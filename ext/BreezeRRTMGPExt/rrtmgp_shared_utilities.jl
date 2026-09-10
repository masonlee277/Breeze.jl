#####
##### Shared utilities for clear-sky and all-sky RRTMGP radiation
#####

using Oceananigans.Operators: ℑzᵃᵃᶠ, Δzᶜᶜᶜ
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: ConstantField
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Utils: launch!

using RRTMGP: RRTMGPSolver
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.VolumeMixingRatios: VmrGM

using Breeze.AtmosphereModels: BackgroundAtmosphere, specific_prognostic_moisture,
                              grid_moisture_fractions
using Breeze.Thermodynamics: MoistureMassFractions, dry_air_mass_fraction

#####
##### Volume mixing ratio initialization (shared by clear-sky and all-sky)
#####

# Zero-initialized global-mean volume mixing ratios: H₂O and O₃ vary per layer
# and column, all other gases are well-mixed scalars.
function initialize_global_mean_vmr(Ngas, Nz, Nc, FT, ArrayType)
    vmr_h2o = ArrayType{FT}(undef, Nz, Nc)
    vmr_o3 = ArrayType{FT}(undef, Nz, Nc)
    global_mean_vmr = ArrayType{FT}(undef, Ngas)

    fill!(vmr_h2o, zero(FT))
    fill!(vmr_o3, zero(FT))
    fill!(global_mean_vmr, zero(FT))

    return VmrGM(vmr_h2o, vmr_o3, global_mean_vmr)
end

#####
##### Gas state update (shared by clear-sky and all-sky)
#####

# Reconstruct from interior pressures in physical height, never dynamical halos.
@inline function rrtmgp_face_pressure(i, j, k, grid, pressure, density, gravity, ::Val{false})
    lower = clamp(k - 1, 1, size(grid, 3) - 1)
    upper = lower + 1
    z_lower = znode(i, j, lower, grid, Center(), Center(), Center())
    z_upper = znode(i, j, upper, grid, Center(), Center(), Center())
    z_face = znode(i, j, k, grid, Center(), Center(), Face())
    @inbounds p_lower = pressure[i, j, lower]
    @inbounds p_upper = pressure[i, j, upper]
    weight = (z_face - z_lower) / (z_upper - z_lower)
    return p_lower + weight * (p_upper - p_lower)
end

# A single layer has no resolved pressure gradient. Retain this supported shape
# with an explicit local hydrostatic boundary closure, not a reconstructed
# hydrostatic column. Its dry molecular amount still uses the cell's actual mass.
@inline function rrtmgp_face_pressure(i, j, k, grid, pressure, density, gravity, ::Val{true})
    z_center = znode(i, j, 1, grid, Center(), Center(), Center())
    z_face = znode(i, j, k, grid, Center(), Center(), Face())
    @inbounds return pressure[i, j, 1] - density[i, j, 1] * gravity * (z_face - z_center)
end

function validate_rrtmgp_gas_columns(as)
    lower_pressure = @view as.p_lev[1:end - 1, :]
    upper_pressure = @view as.p_lev[2:end, :]
    layer_pressure = @view as.layerdata[2, :, :]
    dry_columns = @view as.layerdata[1, :, :]
    valid_pressure = all(p -> isfinite(p) & (p > 0), as.p_lev) &&
                     all(isfinite, layer_pressure) &&
                     all(lower_pressure .> layer_pressure) &&
                     all(layer_pressure .> upper_pressure)
    valid_pressure || throw(ArgumentError(
        "RRTMGP interface pressures must be positive and strictly bracket every layer pressure."))
    all(c -> isfinite(c) & (c > 0), dry_columns) || throw(ArgumentError(
        "RRTMGP requires positive finite physical-layer dry-air column amounts."))
    return nothing
end

function update_rrtmgp_gas_state!(as::AtmosphericState, model, surface_temperature,
                                  background_atmosphere::BackgroundAtmosphere, params)
    grid = model.grid
    arch = architecture(grid)

    # Keep the model's diagnosed center pressures. Reconstruct radiative interface
    # pressures in physical height; a diagnostic field's zero-gradient halos do
    # not locate the physical top/bottom interfaces of its boundary cells.
    p = dynamics_pressure(model.dynamics)
    T = model.temperature
    qᵛᵉ = specific_prognostic_moisture(model)
    ρ = total_density(model.dynamics)

    g = params.grav
    single_layer = Val(size(grid, 3) == 1)
    mᵈ = params.molmass_dryair
    mᵛ = params.molmass_water
    ℕᴬ = params.avogad
    O₃ = background_atmosphere.O₃  # Can be ConstantField or Field

    launch!(arch, grid, :xyz, _update_rrtmgp_gas_state!, as, grid, p, T, qᵛᵉ, ρ,
            model.microphysics, model.microphysical_fields, surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃, single_layer)
    validate_rrtmgp_gas_columns(as)
    return nothing
end

@kernel function _update_rrtmgp_gas_state!(as, grid, p, T, qᵛᵉ, ρ, microphysics, microphysical_fields,
                                         surface_temperature, g, mᵈ, mᵛ, ℕᴬ, O₃, single_layer)
    i, j, k = @index(Global, NTuple)

    Nz = size(grid, 3)
    c = rrtmgp_column_index(i, j, grid.Nx)

    layerdata = as.layerdata
    pᶠ = as.p_lev
    Tᶠ = as.t_lev
    T₀ = as.t_sfc

    vmr_h2o = as.vmr.vmr_h2o
    vmr_o3 = as.vmr.vmr_o3

    @inbounds begin
        # Layer (cell-centered) values
        pᶜ = p[i, j, k]
        q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ[i, j, k],
                                   qᵛᵉ[i, j, k], microphysical_fields)
        qᵛₖ = max(q.vapor, zero(q.vapor))
        # Retain the existing negative-vapor policy, but exclude every condensate
        # reservoir from the dry-air denominator of the moist-total-air fractions.
        qᵈ = dry_air_mass_fraction(MoistureMassFractions(qᵛₖ, q.liquid, q.ice))

        # Reconstruct pressure at the lower physical face; retain the existing
        # face-temperature staging and Planck-source smoothing policy.
        pᶠₖ = rrtmgp_face_pressure(i, j, k, grid, p, ρ, g, single_layer)
        Tᶠₖ = ℑzᵃᵃᶠ(i, j, k, grid, T)
        Tᶠₖ₊₁ = ℑzᵃᵃᶠ(i, j, k+1, grid, T)

        # Use face-averaged temperature for the RRTMGP layer temperature.
        # This ensures consistency between RRTMGP's layer and level Planck sources,
        # preventing 2Δz oscillations in the radiative heating rate that arise when
        # RRTMGP's linear-in-tau source correction amplifies lay_source − lev_source
        # mismatches at the grid Nyquist frequency.
        Tᶜ = (Tᶠₖ + Tᶠₖ₊₁) / 2

        # RRTMGP Planck/source lookup tables are defined over a finite temperature range.
        # Clamp temperatures to avoid extrapolation that can yield tiny negative source values
        # and trigger DomainErrors in geometric means.
        # TODO: This clamping should ideally be done internally in RRTMGP.jl.
        Tmin = 160
        Tmax = 355
        Tᶜ = clamp(Tᶜ, Tmin, Tmax)
        Tᶠₖ = clamp(Tᶠₖ, Tmin, Tmax)

        # Store level values
        pᶠ[k, c] = pᶠₖ
        Tᶠ[k, c] = Tᶠₖ

        # Topmost level (once)
        if k == 1
            pᶠ[Nz + 1, c] = rrtmgp_face_pressure(i, j, Nz + 1, grid, p, ρ, g, single_layer)
            Tᴺ⁺¹ = ℑzᵃᵃᶠ(i, j, Nz+1, grid, T)
            Tᶠ[Nz+1, c] = clamp(Tᴺ⁺¹, Tmin, Tmax)
            T₀[c] = clamp(surface_temperature[i, j, 1], Tmin, Tmax)
        end

        # Integrate the represented dry-air density over this physical layer,
        # exactly as cloud paths integrate condensate density. Using Δp/g would
        # impose hydrostatic balance on compressible states and make gas mass
        # depend on pressure interpolation and boundary halo policies.
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        ρₖ = ρ[i, j, k]
        # Validate factors independently: two negative factors can otherwise
        # produce a positive optical mass. Keep the existing vapor-to-zero
        # policy; this does not replace temperature or trace-gas validation.
        valid_mass = isfinite(ρₖ) & (ρₖ > 0) &
                     isfinite(Δz) & (Δz > 0) &
                     isfinite(q.vapor) &
                     isfinite(qᵈ) & (qᵈ > 0) & (qᵈ <= 1) &
                     isfinite(q.liquid) & (q.liquid >= 0) &
                     isfinite(q.ice) & (q.ice >= 0)
        dry_mass_per_area = ρₖ * qᵈ * Δz
        m⁻²_to_cm⁻² = convert(eltype(pᶜ), 1e4)
        column_dry = dry_mass_per_area / mᵈ * ℕᴬ / m⁻²_to_cm⁻² # (molecules / m²) -> (molecules / cm²)

        # Populate layerdata: (column_dry, pᶜ, Tᶜ, relative_humidity)
        layerdata[1, k, c] = ifelse(valid_mass, column_dry, oftype(column_dry, NaN))
        layerdata[2, k, c] = pᶜ
        layerdata[3, k, c] = Tᶜ
        layerdata[4, k, c] = zero(eltype(Tᶜ))

        # H₂O volume mixing ratio from specific humidity
        r = qᵛₖ / qᵈ
        vmr_h2o[k, c] = r * (mᵈ / mᵛ)

        # O₃ volume mixing ratio - index into field (works for ConstantField or Field)
        vmr_o3[k, c] = O₃[i, j, k]
    end
end

#####
##### Surface boundary conditions (shared by gray, clear-sky and all-sky)
#####

"""
$(TYPEDSIGNATURES)

The scalar behind a surface property that is constant in space and time, or `nothing` when the
property carries no such scalar.

A `ConstantField` is a scalar in a field's clothing — its value cannot change — so it reports the
value it holds. A general `Field` reports `nothing`: it may be rewritten between radiation updates,
so there is no single value to speak of.
"""
surface_fraction_scalar(x::Number) = x
surface_fraction_scalar(x::ConstantField) = surface_fraction_scalar(x.constant)
surface_fraction_scalar(x) = nothing

"""
$(TYPEDSIGNATURES)

Throw an `ArgumentError` for any keyword whose value is a spatially uniform scalar outside ``[0, 1]``.

Emissivity and albedo are fractions, so a scalar outside the unit interval is a user error — an albedo
given in percent, say — worth rejecting at construction rather than carrying into the solver. A
property with no single value (a `Field`, a dataset, `nothing`) passes through, since a check at
construction says nothing about what it holds at the next solve.
"""
function validate_surface_fractions(; kw...)
    for (name, value) in kw
        x = surface_fraction_scalar(value)
        isnothing(x) || 0 <= x <= 1 ||
            throw(ArgumentError("`$name` must lie in [0, 1]; received $x."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Wrap a scalar surface property in a `ConstantField` of the working precision, passing anything
already field-valued through unchanged, so that emissivity and both albedos are uniformly
field-valued whether the user supplied a number, a field, or a dataset.
"""
constant_field_property(x::Number, FT) = ConstantField(convert(FT, x))
constant_field_property(x, FT) = x

"""
$(TYPEDSIGNATURES)

Copy the surface emissivity `ε` and the direct and diffuse albedos `αᵈ`, `αˢ` from
`surface_properties` into RRTMGP's band-by-column boundary-condition arrays `ε₀`, `αᵈ₀`, `αˢ₀`.

Nothing else writes those arrays, so a spatially varying emissivity or albedo would otherwise never
reach the solver, which would read whatever the allocation happened to contain. Call once at
construction and again before every solve, so a property that evolves is picked up rather than frozen.

Breeze treats all three properties as spectrally grey: every band receives the same value.
"""
function update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, surface_properties, grid)
    arch = architecture(grid)

    launch!(arch, grid, :xy, _update_rrtmgp_surface_boundary_conditions!,
            ε₀, αᵈ₀, αˢ₀,
            surface_properties.surface_emissivity,
            surface_properties.direct_surface_albedo,
            surface_properties.diffuse_surface_albedo,
            grid)

    return nothing
end

# Full-spectrum (clear-sky and all-sky) models keep both RTE solvers inside one `RRTMGPSolver`,
# reached through RRTMGP's own accessors rather than its internal field nesting.
update_rrtmgp_surface_boundary_conditions!(solver::RRTMGPSolver, surface_properties, grid) =
    update_rrtmgp_surface_boundary_conditions!(RRTMGP.surface_emissivity(solver),
                                               RRTMGP.direct_sw_surface_albedo(solver),
                                               RRTMGP.diffuse_sw_surface_albedo(solver),
                                               surface_properties, grid)

@kernel function _update_rrtmgp_surface_boundary_conditions!(ε₀, αᵈ₀, αˢ₀, ε, αᵈ, αˢ, grid)
    i, j = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        εᵢⱼ = ε[i, j, 1]
        αᵈᵢⱼ = αᵈ[i, j, 1]
        αˢᵢⱼ = αˢ[i, j, 1]

        for b in 1:size(ε₀, 1)
            ε₀[b, c] = εᵢⱼ
        end

        for b in 1:size(αᵈ₀, 1)
            αᵈ₀[b, c] = αᵈᵢⱼ
            αˢ₀[b, c] = αˢᵢⱼ
        end
    end
end

#####
##### Copy fluxes to Oceananigans fields (shared by clear-sky and all-sky)
#####

function copy_rrtmgp_fluxes_to_fields!(rtm, solver, grid)
    arch = architecture(grid)

    # (Nz+1, Nc) presentation views, refreshed by update_lw_fluxes!/update_sw_fluxes!
    lw_flux_up = RRTMGP.lw_flux_up(solver)
    lw_flux_dn = RRTMGP.lw_flux_dn(solver)
    sw_flux_up = RRTMGP.sw_flux_up(solver)
    sw_flux_dn = RRTMGP.sw_flux_dn(solver)  # Total SW (direct + diffuse)

    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_up = rtm.upwelling_shortwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux

    Nx, Ny, Nz = size(grid)
    launch!(arch, grid, (Nx, Ny, Nz+1), _copy_rrtmgp_fluxes!,
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
            lw_flux_up, lw_flux_dn, sw_flux_up, sw_flux_dn, grid)

    return nothing
end

@kernel function _copy_rrtmgp_fluxes!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
                                      lw_flux_up, lw_flux_dn, sw_flux_up, sw_flux_dn, grid)
    i, j, k = @index(Global, NTuple)

    c = rrtmgp_column_index(i, j, grid.Nx)

    @inbounds begin
        ℐ_lw_up[i, j, k] = lw_flux_up[k, c]
        ℐ_lw_dn[i, j, k] = -lw_flux_dn[k, c]
        ℐ_sw_up[i, j, k] = sw_flux_up[k, c]
        ℐ_sw_dn[i, j, k] = -sw_flux_dn[k, c]
    end
end

#####
##### Compute radiation flux divergence from radiative fluxes
#####

function compute_radiation_flux_divergence!(rtm, grid)
    arch = architecture(grid)
    ℐ_lw_up = rtm.upwelling_longwave_flux
    ℐ_lw_dn = rtm.downwelling_longwave_flux
    ℐ_sw_up = rtm.upwelling_shortwave_flux
    ℐ_sw_dn = rtm.downwelling_shortwave_flux
    flux_div = rtm.flux_divergence
    launch!(arch, grid, :xyz, _compute_radiation_flux_divergence!,
            flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    return nothing
end

@kernel function _compute_radiation_flux_divergence!(flux_div, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, grid)
    i, j, k = @index(Global, NTuple)
    # Net flux at faces k and k+1 (positive upward)
    @inbounds begin
        F_k  = ℐ_lw_up[i, j, k]   + ℐ_lw_dn[i, j, k]   + ℐ_sw_up[i, j, k]   + ℐ_sw_dn[i, j, k]
        F_k1 = ℐ_lw_up[i, j, k+1] + ℐ_lw_dn[i, j, k+1] + ℐ_sw_up[i, j, k+1] + ℐ_sw_dn[i, j, k+1]
    end
    Δz = Δzᶜᶜᶜ(i, j, k, grid)
    # Flux divergence: -dF/dz (positive when flux convergence warms)
    @inbounds flux_div[i, j, k] = -(F_k1 - F_k) / Δz
end

# The constructors accept `surface_temperature = nothing` so that a coupled model can bind
# its interface surface temperature after construction; solving without one is an error.
function assert_bound_surface_temperature(rtm)
    isnothing(rtm.surface_properties.surface_temperature) && throw(ArgumentError(
        "This RadiativeTransferModel has no surface temperature: construct it with " *
        "`surface_temperature = ...`, or bind one before the first radiation update " *
        "(coupled models wire their interface surface temperature automatically)."))
    return nothing
end
