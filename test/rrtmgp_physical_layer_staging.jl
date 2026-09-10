include("setup.jl")

using Test, Breeze, Oceananigans, RRTMGP
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using RRTMGP.AtmosphericStates: AtmosphericState
using RRTMGP.VolumeMixingRatios: VmrGM

@testset "RRTMGP physical dry columns" for FT in (Float32, Float64), layers in (1, 4), cloudy in (false, true)
    faces = layers == 1 ? FT[0, 2000] : FT[0, 100, 400, 1000, 2000]
    grid = RectilinearGrid(default_arch, FT; size=layers, z=faces,
                           topology=(Flat, Flat, Bounded), halo=min(3, layers))
    constants = ThermodynamicConstants(FT)
    microphysics = cloudy ? PredictedParticlePropertiesMicrophysics(FT) : nothing
    dynamics = Breeze.AtmosphereModels.materialize_dynamics(
        CompressibleDynamics(; reference_state=nothing), grid, (;), constants)
    microphysical_fields = Breeze.AtmosphereModels.materialize_microphysical_fields(microphysics, grid, (;))
    model = (; grid, dynamics, microphysics, microphysical_fields, temperature=CenterField(grid))
    density, vapor = FT(1.1), FT(0.01)
    liquid, ice = cloudy ? (FT(0.002), FT(0.001)) : (FT(0), FT(0))
    set!(model.dynamics.total_density, density)
    set!(model.dynamics.dry_density, density * (1 - vapor - liquid - ice))
    gas_constant = constants.molar_gas_constant *
        ((1 - vapor - liquid - ice) / constants.dry_air.molar_mass + vapor / constants.vapor.molar_mass)
    set!(model.microphysical_fields.qᵛ, vapor)
    if cloudy
        set!(model.microphysical_fields.ρqᶜˡ, density * liquid)
        set!(model.microphysical_fields.ρqⁱ, density * ice)
        set!(model.microphysical_fields.ρqᶠ, density * ice / 2)
        set!(model.microphysical_fields.ρbᶠ, density * ice / FT(800))
    end
    set!(model.temperature, FT(280))
    fill_halo_regions!(model.temperature)
    array(dims...) = on_architecture(default_arch, zeros(FT, dims...))
    vmr = VmrGM(array(layers, 1), array(layers, 1), array(1))
    state = AtmosphericState(array(1), array(1), array(4, layers, 1),
        array(layers+1, 1), array(layers+1, 1), array(1), vmr, nothing, nothing)
    parameters = (; grav=constants.gravitational_acceleration,
        molmass_dryair=constants.dry_air.molar_mass,
        molmass_water=constants.vapor.molar_mass, avogad=FT(6.02214076e23))
    background = BackgroundAtmosphere(; O₃=ConstantField(FT(3e-8)))
    extension = Base.get_extension(Breeze, :BreezeRRTMGPExt)
    stage!() = extension.update_rrtmgp_gas_state!(state, model, ConstantField(FT(280)), background, parameters)
    dry = 1 - Float64(vapor) - Float64(liquid) - Float64(ice)
    expected = Float64(density) * dry .* diff(Float64.(faces)) / parameters.molmass_dryair * parameters.avogad / 1e4
    expected_vapor = Float64(vapor) / dry * parameters.molmass_dryair / parameters.molmass_water
    for fraction in (FT(1), FT(0.75))
        slope = fraction * density * parameters.grav
        set!(model.dynamics.pressure, z -> FT(95000) - slope * z)
        set!(model.temperature, z -> (FT(95000) - slope * z) / (density * gas_constant))
        fill_halo_regions!(model.temperature)
        model.dynamics.pressure.data[:, :, 0] .= FT(NaN)
        model.dynamics.pressure.data[:, :, layers+1] .= FT(NaN)
        prior = Array(parent(model.dynamics.pressure))
        stage!()
        @test vec(Array(state.layerdata)[1, :, :]) ≈ expected rtol=64eps(FT)
        @test all(isapprox.(Array(state.vmr.vmr_h2o), expected_vapor; rtol=64eps(FT)))
        expected_slope = layers == 1 ? density * parameters.grav : slope
        center = layers == 1 ? first(Array(interior(model.dynamics.pressure))) : FT(95000)
        expected_levels = layers == 1 ? center .- expected_slope .* (faces .- FT(1000)) : center .- slope .* faces
        @test vec(Array(state.p_lev)) ≈ expected_levels rtol=64eps(FT)
        @test isequal(Array(parent(model.dynamics.pressure)), prior)
    end
    set!(model.dynamics.pressure, FT(-1))
    @test_throws ArgumentError stage!()
    set!(model.dynamics.pressure, z -> FT(95000) - density * parameters.grav * z)
    set!(model.temperature, z -> (FT(95000) - density * parameters.grav * z) / (density * gas_constant))
    fill_halo_regions!(model.temperature)
    stage!()
    @test vec(Array(state.layerdata)[1, :, :]) ≈ expected rtol=64eps(FT)
end
