using Breeze.AtmosphereModels.Diagnostics: Diagnostics
using Breeze.AtmosphereModels: AtmosphereModel, specific_prognostic_moisture, moisture_prognostic_name

using Oceananigans.Fields: Field, set!
using Breeze.Thermodynamics: temperature, LiquidIceDensityState, mixture_gas_constant
using Breeze.BoundaryConditions: theta_to_energy_bcs, materialize_atmosphere_field_bcs,
    map_field_boundary_conditions, set_energy_flux_response

const PotentialTemperatureModel = AtmosphereModel{<:Any, <:LiquidIcePotentialTemperatureFormulation}

AtmosphereModels.specific_thermodynamic_field(formulation::LiquidIcePotentialTemperatureFormulation) = formulation.potential_temperature

#####
##### Helper accessors
#####

AtmosphereModels.liquid_ice_potential_temperature_density(model::PotentialTemperatureModel) = model.formulation.potential_temperature_density
AtmosphereModels.liquid_ice_potential_temperature(model::PotentialTemperatureModel) = model.formulation.potential_temperature
AtmosphereModels.static_energy(model::PotentialTemperatureModel) = Diagnostics.StaticEnergy(model, :specific)

"""
$(TYPEDSIGNATURES)

Return the static energy density as a `Field` with boundary conditions that return
energy fluxes when used with `BoundaryConditionOperation`.

For `LiquidIcePotentialTemperatureFormulation`, the prognostic variable is potential
temperature density `ρθ`. This function converts the `ρθ` boundary conditions to
energy flux boundary conditions using the inverse physical heat response of the
implemented temperature coordinate.
"""
function AtmosphereModels.static_energy_density(model::PotentialTemperatureModel)
    ρθ = model.formulation.potential_temperature_density
    ρθ_bcs = ρθ.boundary_conditions

    # Convert θ BCs to energy BCs
    ρs_bcs = theta_to_energy_bcs(ρθ_bcs)
    ρs_bcs = map_field_boundary_conditions(set_energy_flux_response, ρs_bcs,
        model.formulation, model.dynamics, model.grid, model.microphysics)

    # Regularize the converted BCs (populate microphysics, constants, side)
    loc = (Center(), Center(), Center())
    ρs_bcs = materialize_atmosphere_field_bcs(ρs_bcs, loc, model.grid, model.dynamics, model.microphysics,
                                              nothing, model.thermodynamic_constants, nothing, nothing, nothing)

    # Create the energy density operation and wrap in a Field with proper BCs
    ρs_op = Diagnostics.StaticEnergy(model, :density)
    return Field(ρs_op; boundary_conditions=ρs_bcs)
end

#####
##### Tendency computation
#####

# Convert physical radiative heating Q [W m^-3] into the numerator of the
# existing theta tendency. Reference-pressure states retain their old formula.
# Equilibrium-moisture (:ρqᵉ) also retains the LEGACY approximation: its phase
# partition changes algebraically with theta, and the fixed-phase Jacobian below
# does not establish energy consistency for saturation adjustment.
@inline radiative_heat_source(Q, state, constants, ρᵈ, ρ, T, cᵖᵐ, Π, ::Val) = Q

# The current :ρqᵛ implementations store actual vapor and either independent
# phase reservoirs or no condensate; their thermodynamic diagnostic adjustment
# is the identity (P3, non-equilibrium bulk schemes, Kessler, or no microphysics).
# A split microphysical process update is distinct from this instantaneous source.
# For fixed phases and density, the implemented linear theta definition gives
#   theta = (T - L) / Pi,  L = (L_l q_l + L_i q_i) / cp,
#   dtheta/dT = [1 - kappa + kappa L/T] / Pi.
# Combining rho*cv*dT/dt = Q and the dry-coupled prognostic rho_d*theta yields
#   d(rho_d*theta)/dt = (rho_d/rho)*(1 + R*L/(cv*T))*Q/(cp*Pi).
# This uses the current diagnosed T, avoiding a second Newton solve. Newton and
# FixedIterations are assumed to approximate the implicit root to their stated
# accuracy. The intentionally non-iterated option has its own derivative below.
# No change is made to user Fρs, whose existing forcing carrier is rho_d, not rho.
@inline function radiative_heat_source(Q, state::LiquidIceDensityState, constants,
                                     ρᵈ, ρ, T, cᵖᵐ, Π, ::Val{:ρqᵛ})
    q = state.moisture_mass_fractions
    Rᵐ = mixture_gas_constant(q, constants)
    cᵛᵐ = cᵖᵐ - Rᵐ
    L = (constants.liquid.reference_latent_heat * q.liquid +
         constants.ice.reference_latent_heat * q.ice) / cᵖᵐ
    factor = radiative_density_factor(state.temperature_solver, state, T, Rᵐ, cᵛᵐ, L, Π)
    return (ρᵈ / ρ) * factor * Q
end

@inline radiative_density_factor(solver, state, T, Rᵐ, cᵛᵐ, L, Π) = 1 + Rᵐ * L / (cᵛᵐ * T)

# The explicitly approximate inversion is T = Tdry(theta) + L, Tdry ∝ theta^gamma.
# Its physical heat response requires d(rho_d*theta)/dt = (rho_d/rho)*Q*theta/[cp*(T-L)].
# This couples heating to that actual inversion; it does not make it an exact
# solution of the cloudy implicit root or change the chosen temperature solver.
@inline function radiative_density_factor(::Nothing, state, T, Rᵐ, cᵛᵐ, L, Π)
    return Π * state.potential_temperature / (T - L)
end

function AtmosphereModels.compute_thermodynamic_tendency!(model::PotentialTemperatureModel, common_args)
    grid = model.grid
    arch = grid.architecture

    ρθ_args = (
        Val(1),
        model.forcing.ρθ,
        model.forcing.ρs,
        model.advection.ρθ,
        radiation_flux_divergence(model.radiation),
        common_args...)

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, compute_potential_temperature_tendency!, Gρθ, grid, ρθ_args)
    return nothing
end

@inline function potential_temperature_tendency(i, j, k, grid,
                                                id,
                                                ρθ_forcing,
                                                ρs_forcing,
                                                advection,
                                                radiation_flux_divergence_field,
                                                dynamics,
                                                formulation::LiquidIcePotentialTemperatureFormulation,
                                                constants,
                                                specific_prognostic_moisture,
                                                velocities,
                                                microphysics,
                                                microphysical_fields,
                                                closure,
                                                closure_fields,
                                                clock,
                                                model_fields)

    potential_temperature = formulation.potential_temperature
    ρ_field = dynamics_density(dynamics)                # coupling density ρᵈ (advection/diffusion carrier)
    @inbounds ρ = total_density(dynamics)[i, j, k]  # total ρ (mass fractions)
    @inbounds qᵛᵉ = specific_prognostic_moisture[i, j, k]

    # Compute moisture fractions first
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)

    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    closure_buoyancy = AtmosphereModelBuoyancy(dynamics, formulation, constants)

    Fρs = ρs_forcing(i, j, k, grid, clock, model_fields)
    div_ℐ = radiation_flux_divergence(i, j, k, grid, radiation_flux_divergence_field)
    @inbounds begin
        ρᵈ = ρ_field[i, j, k]
        T = model_fields.T[i, j, k]
    end
    Qθ = radiative_heat_source(div_ℐ, 𝒰, constants, ρᵈ, ρ, T, cᵖᵐ, Π, Val(moisture_prognostic_name(microphysics)))

    return ( - div_ρUc(i, j, k, grid, advection, ρ_field, velocities, potential_temperature)
             + c_div_ρU(i, j, k, grid, dynamics, velocities, potential_temperature)
             - ∇_dot_Jᶜ(i, j, k, grid, ρ_field, closure, closure_fields, id, potential_temperature, clock, model_fields, closure_buoyancy)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + (Fρs + Qθ) / (cᵖᵐ * Π)
    )
end

#####
##### Set thermodynamic variables
#####

AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Union{Val{:ρθ}, Val{:ρθˡⁱ}}, value) =
    set!(model.formulation.potential_temperature_density, value)

function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Union{Val{:θ}, Val{:θˡⁱ}}, value)
    set!(model.formulation.potential_temperature, value)
    ρ = dynamics_density(model.dynamics)
    θˡⁱ = model.formulation.potential_temperature
    set!(model.formulation.potential_temperature_density, ρ * θˡⁱ)
    return nothing
end

# Setting from static energy
function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:s}, value)
    formulation = model.formulation
    s = model.temperature # scratch space
    set!(s, value)

    grid = model.grid
    arch = grid.architecture
    launch!(arch, grid, :xyz,
            _potential_temperature_from_energy!,
            formulation.potential_temperature_density,
            formulation.potential_temperature,
            grid,
            s,
            specific_prognostic_moisture(model),
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:ρs}, value)
    ρs = model.temperature # scratch space
    set!(ρs, value)
    ρ = dynamics_density(model.dynamics)
    return set_thermodynamic_variable!(model, Val(:s), ρs / ρ)
end

@kernel function _potential_temperature_from_energy!(potential_temperature_density,
                                                     potential_temperature,
                                                     grid,
                                                     specific_energy,
                                                     specific_prognostic_moisture,
                                                     dynamics,
                                                     microphysics,
                                                     microphysical_fields,
                                                     constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        pᵣ = dynamics_pressure(dynamics)[i, j, k]
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρθ = ρᵈθ)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        s = specific_energy[i, j, k]
    end

    z = znode(i, j, k, grid, c, c, c)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    𝒰s₀ = StaticEnergyState(s, q, z, pᵣ)
    𝒰s₁ = maybe_adjust_thermodynamic_state(𝒰s₀, microphysics, qᵛᵉ, constants)
    T = temperature(𝒰s₁, constants)

    pˢᵗ = standard_pressure(dynamics)
    q₁ = 𝒰s₁.moisture_mass_fractions
    𝒰θ = LiquidIcePotentialTemperatureState(zero(T), q₁, pˢᵗ, pᵣ)
    𝒰θ = with_temperature(𝒰θ, T, constants)
    θ = 𝒰θ.potential_temperature
    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵈ * θ
end

#####
##### Setting temperature directly
#####

"""
    $(TYPEDSIGNATURES)

Set the thermodynamic state from in-situ temperature ``T``.

The temperature is converted to liquid-ice potential temperature `θˡⁱ` using
the relation between ``T`` and `θˡⁱ`` that accounts for the moisture distribution.

For unsaturated air (no condensate), this simplifies to ``θ = T / Π`` where
``Π`` is the Exner function.
"""
function AtmosphereModels.set_thermodynamic_variable!(model::PotentialTemperatureModel, ::Val{:T}, value)
    T_field = model.temperature # use temperature field as scratch/storage
    set!(T_field, value)

    grid = model.grid
    arch = grid.architecture
    formulation = model.formulation

    launch!(arch, grid, :xyz,
            _potential_temperature_from_temperature!,
            formulation.potential_temperature_density,
            formulation.potential_temperature,
            grid,
            T_field,
            specific_prognostic_moisture(model),
            model.dynamics,
            model.microphysics,
            model.microphysical_fields,
            model.thermodynamic_constants)

    return nothing
end

@kernel function _potential_temperature_from_temperature!(potential_temperature_density,
                                                          potential_temperature,
                                                          grid,
                                                          temperature_field,
                                                          specific_prognostic_moisture,
                                                          dynamics,
                                                          microphysics,
                                                          microphysical_fields,
                                                          constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = total_density(dynamics)[i, j, k]      # total ρ (mass fractions)
        ρᵈ = dynamics_density(dynamics)[i, j, k]  # coupling density ρᵈ (ρθ = ρᵈθ)
        qᵛᵉ = specific_prognostic_moisture[i, j, k]
        T = temperature_field[i, j, k]
    end

    # Get moisture fractions (vapor only for unsaturated air)
    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, microphysical_fields)
    pᵣ = pressure_from_density_temperature(i, j, k, grid, dynamics, ρ, T, q, constants)

    # Convert temperature to potential temperature using the inverse of the T(θ) relation
    pˢᵗ = standard_pressure(dynamics)
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), q, pˢᵗ, pᵣ)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    θ = 𝒰₁.potential_temperature

    @inbounds potential_temperature[i, j, k] = θ
    @inbounds potential_temperature_density[i, j, k] = ρᵈ * θ
end
