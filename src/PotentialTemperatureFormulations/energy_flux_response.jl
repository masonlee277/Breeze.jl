using Breeze.BoundaryConditions: BoundaryConditions as BreezeBoundaryConditions

# Physical heat must use the same coordinate and temperature inversion as the
# volumetric radiation tendency. These immutable callbacks retain only metadata;
# prognostic masses and temperature are read at the current boundary call.
struct ReferencePressureEnergyFluxResponse{P, F, M}
    standard_pressure :: P
    reference_pressure :: F
    moisture_name :: M
end

struct DensityEnergyFluxResponse{P, S, M}
    standard_pressure :: P
    temperature_solver :: S
    moisture_name :: M
end

Adapt.adapt_structure(to, r::ReferencePressureEnergyFluxResponse) =
    ReferencePressureEnergyFluxResponse(adapt(to, r.standard_pressure),
        adapt(to, r.reference_pressure), r.moisture_name)
Adapt.adapt_structure(to, r::DensityEnergyFluxResponse) =
    DensityEnergyFluxResponse(adapt(to, r.standard_pressure),
        adapt(to, r.temperature_solver), r.moisture_name)

BreezeBoundaryConditions.convert_energy_to_theta_bcs(bcs, ::LiquidIcePotentialTemperatureFormulation, constants) =
    BreezeBoundaryConditions.convert_energy_to_theta_bcs(bcs, Val(:LiquidIcePotentialTemperature), constants)

AtmosphereModels.energy_flux_response(::Union{Val{:LiquidIcePotentialTemperature}, Val{:θ}}, args...) =
    AtmosphereModels.energy_flux_response(LiquidIcePotentialTemperatureFormulation(), args...)

function AtmosphereModels.energy_flux_response(::LiquidIcePotentialTemperatureFormulation, dynamics, grid, microphysics)
    return ReferencePressureEnergyFluxResponse(eltype(grid)(standard_pressure(dynamics)),
        dynamics_pressure(dynamics), Val(moisture_prognostic_name(microphysics)))
end

@inline energy_moisture_field(fields, ::Val{name}) where name = getproperty(fields, name)

@inline function (r::ReferencePressureEnergyFluxResponse)(i, j, k, grid, ef, Q, fields)
    @inbounds begin
        ρ = ef.density[i, j, k]
        p = r.reference_pressure[i, j, k]
        θ = fields.ρθ[i, j, k] / ρ
        qᵛᵉ = energy_moisture_field(fields, r.moisture_name)[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, ef.microphysics, ρ, qᵛᵉ, fields)
    state = LiquidIcePotentialTemperatureState(θ, q, r.standard_pressure, p)
    cp = mixture_heat_capacity(q, ef.thermodynamic_constants)
    Π = exner_function(state, ef.thermodynamic_constants)
    return Q / (cp * Π)
end

@inline function (r::DensityEnergyFluxResponse)(i, j, k, grid, ef, Q, fields)
    moisture = energy_moisture_field(fields, r.moisture_name)
    ρ = total_density(i, j, k, fields.ρᵈ, ef.microphysics, moisture, fields)
    @inbounds begin
        ρᵈ = fields.ρᵈ[i, j, k]
        θ = fields.ρθ[i, j, k] / ρᵈ
        T = fields.T[i, j, k]
        qᵛᵉ = moisture[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, grid, ef.microphysics, ρ, qᵛᵉ, fields)
    state = LiquidIceDensityState(θ, q, r.standard_pressure, ρ, r.temperature_solver)
    cp = mixture_heat_capacity(q, ef.thermodynamic_constants)
    R = mixture_gas_constant(q, ef.thermodynamic_constants)
    Π = (ρ * R * T / r.standard_pressure)^(R / cp)
    return radiative_heat_source(Q, state, ef.thermodynamic_constants, ρᵈ, ρ, T, cp, Π, r.moisture_name) / (cp * Π)
end
