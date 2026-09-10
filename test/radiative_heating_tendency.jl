include("setup.jl")

using Test
using Breeze
using Oceananigans: Oceananigans, RectilinearGrid, Flat, Bounded, CenterField, set!, fields, prognostic_fields
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: update_state!

const HeatingAM = Breeze.AtmosphereModels

struct PrescribedVolumetricHeating{F}
    flux_divergence :: F
end
HeatingAM.update_radiation!(::PrescribedVolumetricHeating, model) = nothing

heating_array(field) = copy(Array(interior(field)))
heating_prognostics(model) = map(heating_array, prognostic_fields(model))

function heating_model(FT; compressible=true, phase=:moist, Q=0, Fρs=0, radiation=true, solver=:default,
                       boundary_conditions=(;))
    grid = RectilinearGrid(default_arch, FT; size=4, z=(0, 400), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    dynamics = compressible ? CompressibleDynamics(ExplicitTimeStepping();
        surface_pressure=FT(1e5), standard_pressure=FT(1e5)) :
        AnelasticDynamics(ReferenceState(grid, constants;
            surface_pressure=FT(1e5), potential_temperature=FT(280)))
    microphysics = phase == :p3 ? PredictedParticlePropertiesMicrophysics(FT) :
                   phase == :equilibrium ? SaturationAdjustment(FT) : nothing
    temperature_solver = solver == :none ? nothing : solver == :fixed ? FixedIterations(6) :
                         HeatingAM.DefaultTemperatureSolver()
    formulation = LiquidIcePotentialTemperatureFormulation(; temperature_solver)
    heat = CenterField(grid)
    set!(heat, FT(Q))
    model = AtmosphereModel(grid; dynamics, formulation, thermodynamic_constants=constants,
        microphysics, boundary_conditions, scalar_advection=nothing, momentum_advection=nothing,
        forcing=(; ρs=Returns(FT(Fρs))),
        radiation=radiation ? PrescribedVolumetricHeating(heat) : nothing)
    density = compressible ? (; ρᵈ=FT(1.1)) : (;)
    if phase == :p3
        # Fixed, admissible prognostic populations. Neither a microphysics step
        # nor a physical claim that this mixed population remains frozen follows.
        set!(model; density..., T=FT(280), qᵛ=FT(0.006), qᶜˡ=FT(0.001), qʳ=FT(0.0002),
            nʳ=FT(1e5), qⁱ=FT(0.0005), nⁱ=FT(1e4), qᶠ=FT(0.0001),
            bᶠ=FT(0.0001/400), qʷⁱ=FT(0.0001), enforce_mass_conservation=false)
    elseif phase == :equilibrium
        set!(model; density..., T=FT(280), qᵉ=FT(0.02), enforce_mass_conservation=false)
    else
        set!(model; density..., T=FT(280), qᵛ=phase == :dry ? zero(FT) : FT(0.02),
            enforce_mass_conservation=false)
    end
    update_state!(model; compute_tendencies=false)
    return model, heat
end

function heating_tendency(model)
    common = (model.dynamics, model.formulation, model.thermodynamic_constants,
        HeatingAM.specific_prognostic_moisture(model), HeatingAM.transport_velocities(model),
        model.microphysics, model.microphysical_fields, model.closure, model.closure_fields,
        model.clock, fields(model))
    HeatingAM.compute_thermodynamic_tendency!(model, common)
    return heating_array(model.timestepper.Gⁿ.ρθ)
end

function heating_coefficients(model, phase, compressible)
    # Independent arithmetic from represented diagnostic/raw fields. Coating is
    # liquid; rime mass is already inside total ice and is not counted twice.
    ρ = Float64.(heating_array(HeatingAM.total_density(model.dynamics)))
    ρd = Float64.(heating_array(HeatingAM.dynamics_density(model.dynamics)))
    T = Float64.(heating_array(model.temperature))
    μ = model.microphysical_fields
    if phase == :equilibrium
        qv = Float64.(heating_array(μ.qᵛ))
        ql = Float64.(heating_array(μ.qˡ))
        qi = hasproperty(μ, :qⁱ) ? Float64.(heating_array(μ.qⁱ)) : zeros(size(T))
    else
        qv = Float64.(heating_array(HeatingAM.specific_prognostic_moisture(model)))
        ql = phase == :p3 ? (Float64.(heating_array(μ.ρqᶜˡ)) .+
            Float64.(heating_array(μ.ρqʳ)) .+ Float64.(heating_array(μ.ρqʷⁱ))) ./ ρ : zeros(size(T))
        qi = phase == :p3 ? Float64.(heating_array(μ.ρqⁱ)) ./ ρ : zeros(size(T))
    end
    c = model.thermodynamic_constants
    qd = 1 .- qv .- ql .- qi
    cp = qd .* c.dry_air.heat_capacity .+ qv .* c.vapor.heat_capacity .+
         ql .* c.liquid.heat_capacity .+ qi .* c.ice.heat_capacity
    Rd = Float64(c.molar_gas_constant) / Float64(c.dry_air.molar_mass)
    Rv = Float64(c.molar_gas_constant) / Float64(c.vapor.molar_mass)
    R = qd .* Rd .+ qv .* Rv
    cv = cp .- R
    L = (c.liquid.reference_latent_heat .* ql .+ c.ice.reference_latent_heat .* qi) ./ cp
    p = compressible ? ρ .* R .* T : Float64.(heating_array(HeatingAM.dynamics_pressure(model.dynamics)))
    Π = (p ./ HeatingAM.standard_pressure(model.dynamics)) .^ (R ./ cp)
    D = 1 .- R ./ cp .+ (R ./ cp) .* L ./ T
    θ = Float64.(heating_array(model.formulation.potential_temperature))
    return (; ρ, ρd, T, cp, cv, Π, D, L, θ)
end

@testset "Density-state heat Jacobian, scalar $FT" for FT in (Float32, Float64)
    thermo = Breeze.Thermodynamics
    formulation = Breeze.PotentialTemperatureFormulations
    constants = ThermodynamicConstants(FT)
    for solver in (NewtonSolver(FT), FixedIterations(6), nothing),
        q in (thermo.MoistureMassFractions(FT(0.02)),
              thermo.MoistureMassFractions(FT(0.006), FT(0.0013), FT(0.0005)))
        ρ = FT(1.2)
        ρd = ρ * (1 - q.vapor - q.liquid - q.ice)
        state = thermo.LiquidIceDensityState(zero(FT), q, FT(1e5), ρ, solver)
        state = thermo.with_temperature(state, FT(280), constants)
        T = thermo.temperature(state, constants)
        cp = thermo.mixture_heat_capacity(q, constants)
        Π = thermo.exner_function(state, constants)
        # Widen represented input components BEFORE this independent derivative.
        qv, ql, qi = Float64(q.vapor), Float64(q.liquid), Float64(q.ice)
        qd = 1 - qv - ql - qi
        cp_ref = qd * constants.dry_air.heat_capacity + qv * constants.vapor.heat_capacity +
                 ql * constants.liquid.heat_capacity + qi * constants.ice.heat_capacity
        R = qd * (Float64(constants.molar_gas_constant) / constants.dry_air.molar_mass) +
            qv * (Float64(constants.molar_gas_constant) / constants.vapor.molar_mass)
        cv = cp_ref - R
        L = (constants.liquid.reference_latent_heat * ql + constants.ice.reference_latent_heat * qi) / cp_ref
        Π_ref = (Float64(ρ) * R * Float64(T) / Float64(state.standard_pressure))^(R / cp_ref)
        for Q in (FT(-16), FT(16))
            actual = formulation.radiative_heat_source(Q, state, constants, ρd, ρ, T, cp, Π, Val(:ρqᵛ)) / (cp * Π)
            expected = solver === nothing ? Float64(ρd)/ρ * Q * Float64(state.potential_temperature) /
                (cp_ref * (Float64(T) - L)) : Float64(ρd)/ρ * Q *
                (1 - R/cp_ref + R/cp_ref * L/Float64(T)) / (cv * Π_ref)
            @test isapprox(actual, expected; rtol=128eps(FT), atol=0)
            @test formulation.radiative_heat_source(Q, state, constants, ρd, ρ, T, cp, Π, Val(:ρqᵉ)) == Q
        end
    end
end

@testset "Physical radiative heat to dry-coupled theta, $FT" for FT in (Float32, Float64)
    cases = [(compressible, phase, :default) for compressible in (false, true) for phase in (:dry, :moist, :p3)]
    append!(cases, [(true, :p3, :fixed), (true, :p3, :none)])
    for (compressible, phase, solver) in cases
        model, heat = heating_model(FT; compressible, phase, solver)
        prior = heating_prognostics(model)
        clock = deepcopy(model.clock)
        base = heating_tendency(model)
        coefficients = heating_coefficients(model, phase, compressible)
        c = coefficients
        @test all(isfinite, c.T) && all(>(0), c.cv)
        if compressible && phase != :dry
            @test all(c.ρd .< c.ρ)
        end
        for Q in (FT(-16), FT(16))
            set!(heat, Q)
            actual = Float64.(heating_tendency(model)) .- Float64.(base)
            expected = !compressible ? Q ./ (c.cp .* c.Π) : solver == :none ?
                (c.ρd ./ c.ρ) .* Q .* c.θ ./ (c.cp .* (c.T .- c.L)) :
                (c.ρd ./ c.ρ) .* Q .* c.D ./ (c.cv .* c.Π)
            @test all(isapprox.(actual, expected; rtol=128eps(FT), atol=0))
            if compressible && phase != :dry
                legacy = Q ./ (c.cp .* c.Π)
                @test maximum(abs.(expected .- legacy)) > 1000eps(FT) * maximum(abs, expected)
            end
        end
        @test isequal(heating_prognostics(model), prior)
        @test model.clock.time == clock.time && model.clock.iteration == clock.iteration
        if compressible && phase == :p3 && solver != :none
            @test all(>(0), c.L)
            # The dry/total factor alone omits the cloudy fixed-phase Jacobian.
            partial = (c.ρd ./ c.ρ) ./ (c.cp .* c.Π)
            exact = (c.ρd ./ c.ρ) .* c.D ./ (c.cv .* c.Π)
            @test maximum(abs.(exact .- partial)) > 1000eps(FT) * maximum(abs, exact)
        end
        if compressible
            # Real owner inversion with all mass/phase fields held fixed. Two
            # source-only Euler increments verify the local heat-to-T response.
            set!(heat, FT(1000))
            G = heating_tendency(model) .- base
            θρ = heating_array(model.formulation.potential_temperature_density)
            derivative = 1000 ./ (c.ρ .* c.cv)
            for h in (FT(0.25), FT(0.125))
                set!(model.formulation.potential_temperature_density, θρ .+ h .* G)
                HeatingAM.compute_auxiliary_dynamics_variables!(model)
                response = (Float64.(heating_array(model.temperature)) .- c.T) ./ h
                # Explicit storage-roundoff plus first-order Euler curvature
                # allowance; the instantaneous G test above uses the strict oracle.
                bound = 8eps(FT) .* c.T ./ h .+ 4h .* derivative.^2 ./ c.T
                @test all(abs.(response .- derivative) .<= bound)
            end
            set!(model.formulation.potential_temperature_density, θρ)
            HeatingAM.compute_auxiliary_dynamics_variables!(model)
            @test isequal(heating_prognostics(model), prior)
        end
    end
end

@testset "Legacy forcing and equilibrium radiation scope, $FT" for FT in (Float32, Float64)
    for compressible in (false, true)
        model, heat = heating_model(FT; compressible, phase=:moist, Fρs=7)
        c = heating_coefficients(model, :moist, compressible)
        expected_forcing = 7 ./ (c.cp .* c.Π)
        @test all(isapprox.(Float64.(heating_tendency(model)), expected_forcing; rtol=128eps(FT), atol=0))
        no_radiation, _ = heating_model(FT; compressible, phase=:moist, Fρs=7, radiation=false)
        @test isequal(heating_tendency(model), heating_tendency(no_radiation))
    end
    # No blanket rejection of previously supported equilibrium models. This is
    # a compatibility test ONLY: it does not qualify saturated energy consistency.
    equilibrium, heat = heating_model(FT; phase=:equilibrium)
    @test HeatingAM.moisture_prognostic_name(equilibrium.microphysics) == :ρqᵉ
    @test any(>(0), heating_array(equilibrium.microphysical_fields.qˡ))
    c = heating_coefficients(equilibrium, :equilibrium, true)
    baseline = heating_tendency(equilibrium)
    set!(heat, FT(16))
    actual = Float64.(heating_tendency(equilibrium)) .- Float64.(baseline)
    @test all(isapprox.(actual, 16 ./ (c.cp .* c.Π); rtol=128eps(FT), atol=0))
end

using Oceananigans.BoundaryConditions: FieldBoundaryConditions, FluxBoundaryCondition, getbc
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

@kernel function _surface_flux!(result, bc, grid, clock, model_fields)
    i, j = @index(Global, NTuple)
    @inbounds result[i, j, 1] = getbc(bc, i, j, grid, clock, model_fields)
end

function surface_flux(model, bc)
    result = CenterField(model.grid)
    launch!(default_arch, model.grid, :xy, _surface_flux!, result, bc,
            model.grid, model.clock, fields(model))
    return first(heating_array(result))
end

@testset "Surface energy flux uses the implemented temperature coordinate" begin
    for FT in (Float32,Float64), compressible in (false,true), phase in (:dry,:p3),
        solver in (compressible ? (:default,:fixed,:none) : (:default,)), key in (:ρθ,:ρs)
        Q = FT(100000) # Large source-only probe makes storage rounding measurable.
        bc = key === :ρθ ? Breeze.EnergyFluxBoundaryCondition(Q) : FluxBoundaryCondition(Q)
        boundary_conditions = NamedTuple{(key,)}((FieldBoundaryConditions(bottom=bc),))
        model,_ = heating_model(FT;compressible,phase,solver,
            radiation=false,boundary_conditions)
        c = heating_coefficients(model,phase,compressible)
        before = heating_prognostics(model)
        field = model.formulation.potential_temperature_density
        actual = surface_flux(model, field.boundary_conditions.bottom)
        expected = !compressible ? Q/(c.cp[1]*c.Π[1]) : solver === :none ?
            (c.ρd[1]/c.ρ[1])*Q*c.θ[1]/(c.cp[1]*(c.T[1]-c.L[1])) :
            (c.ρd[1]/c.ρ[1])*Q*c.D[1]/(c.cv[1]*c.Π[1])
        @test isapprox(actual,expected;rtol=128eps(FT),atol=0)
        @test heating_prognostics(model) == before

        energy = Breeze.AtmosphereModels.static_energy_density(model)
        recovered = surface_flux(model, energy.boundary_conditions.bottom)
        @test recovered == Q # Wrapped physical flux is returned directly.
        theta_bcs = (; ρθ=FieldBoundaryConditions(bottom=FluxBoundaryCondition(actual)))
        theta_model,_ = heating_model(FT;compressible,phase,solver,
            radiation=false,boundary_conditions=theta_bcs)
        theta_energy = Breeze.AtmosphereModels.static_energy_density(theta_model)
        inverse = surface_flux(theta_model, theta_energy.boundary_conditions.bottom)
        @test isapprox(inverse,Q;rtol=128eps(FT),atol=0)

        # A unit-area lower face heats a 100-m-thick first cell. Keep all phase
        # masses fixed, apply just this source, and use the owner's real inversion.
        initial = heating_array(field)
        derivative = Float64(Q)/(100*c.ρ[1]*(compressible ? c.cv[1] : c.cp[1]))
        for h in FT.((0.25,0.125))
            perturbed = copy(initial)
            perturbed[1,1,1] += h*actual/FT(100)
            set!(field, perturbed)
            Oceananigans.TimeSteppers.update_state!(model;compute_tendencies=false)
            response = (Float64(first(heating_array(model.temperature)))-c.T[1])/h
            bound = 8eps(FT)*c.T[1]/h + 4h*derivative^2/c.T[1]
            @test abs(response-derivative) <= bound
        end
        set!(field,initial)
        Oceananigans.TimeSteppers.update_state!(model;compute_tendencies=false)
        @test heating_prognostics(model) == before
    end
end
