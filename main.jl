using Pkg
Pkg.activate(".")

using Plots
using StaticArrays
using LinearAlgebra
using Accessors
using Zygote
using StructArrays
using ProgressMeter
using BenchmarkTools
using WriteVTK

struct Particle{T}
    m::T
    ρ::T # density field
    p::T
    r::SVector{3,T}
    v::SVector{3,T} # velocity field
    f::SVector{3,T}
end

function χ(p::Bool) # Indicator function
    if p
        return 1.0
    else
        return 0.0
    end
end

function W_poly6(r, h) # kernel function
    r_norm = norm(r)
    return 315 / (64 * pi * h^9) * (h^2 - r_norm^2)^3 * χ(0 <= r_norm <= h)
end

function ∇2_W_viscosity(r, h::Float64)
    r_norm = norm(r)
    45 / (pi * h^6) * (h - r_norm)
end

function color_smooth(r::Tuple{T,T,T}, particles, h) where {T}
    color_smooth(SVector(r), particles, h)
end
function pressure_smooth(r::Tuple{T,T,T}, particles, h) where {T}
    pressure_smooth(SVector(r), particles, h)
end
function ρ_smooth(r::Tuple{T,T,T}, particles, h) where {T}
    ρ_smooth(SVector(r), particles, h)
end

function color_smooth(r::SVector{3,T}, particles, h) where {T}# smoothed color field 
    sum = 0.0
    for p in particles
        sum += p.m * (1 / p.ρ) * W_poly6(norm(r - p.r), h)
    end
    return sum
end

function ρ_smooth(r::SVector{3,T}, particles, h) where {T}
    sum = 0.0
    for p in particles
        sum += p.m * W_poly6(norm(r - p.r), h)
    end
    return sum
end

function pressure_smooth(r::SVector{3,T}, particles, h) where {T}
    sum = 0.0
    for p in particles
        sum += p.m * p.p / p.ρ * W_poly6(norm(r - p.r), h)
    end
    return sum
end

function eval_pressure_force(particles, i, h::Float64)
    f_pressure = SVector(0, 0, 0)
    for (j, particle_j) in enumerate(particles)
        f_pressure = -particle_j.m * (particles[i].p + particle_j.p) / (2 * particle_j.ρ) * gradient(r -> W_poly6(r, h), particles[i].r - particle_j.r)[1]
    end
    return f_pressure
end

function eval_viscosity_force(particles, i, μ, h)
    f_viscosity = SVector(0, 0, 0)
    for (j, _) in enumerate(particles)
        f_viscosity += particles.m[j] * (particles.v[j] - particles.v[i]) / particles.ρ[j] * ∇2_W_viscosity(particles.r[i] - particles.r[j], h)
    end
    f_viscosity *= μ
    return f_viscosity
end

function apply_gravity()
    return 9.8 * [0.0, 0.0, -1.0]
end

function check_collusion(particle)
    if particle.r[1] < 0
        @reset particle.v[1] = -particle.v[1]
    end
    if particle.r[3] < 0
        @reset particle.v[3] = -particle.v[3]
    end

    return particle
end

# init 
function main()
    k = 1.0 # gas constant
    μ = 0.00001 # viscosity constant
    g = 0
    x_length = 1.0
    z_length = 1.0
    h = 0.05
    num_particles = length(0:2h:x_length) * length(0:2h:z_length)

    particles = begin
        m = fill(8h^3, num_particles)
        ρ = fill(1.0, num_particles)
        p = fill(1.0, num_particles)
        r = [SVector(abs(x + h * randn()), 0, abs(z + h * randn())) for x in 0:2h:x_length for z in 0:2h:z_length]
        v = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]
        f = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]
        StructArray{Particle}((m, ρ, p, r, v, f))
    end

    dt = 0.01
    x, y, z = 0:0.01:3, 0.0, -0.1:0.01:1.4
    times = range(0, 1, step=dt)

    pressure = zeros(num_particles)
    viscosity = zeros(num_particles)

    l = @layout([a; b])
    anim = Animation()

    for (n, time) ∈ enumerate(times)
        @reset particles.ρ = ρ_smooth.(particles.r, [particles], h) #바뀌는 도중에 업데이트 되나? 안될거 같긴 함.

        for (i, particle) in enumerate(particles)
            @reset particles.f[i] = eval_pressure_force(particles, i, h) + eval_viscosity_force(particles, i, μ, h)
            pressure[i] = norm(eval_pressure_force(particles, i, h))
            viscosity[i] = norm(eval_viscosity_force(particles, i, μ, h))
        end
        @reset particles.v .+= (particles.f ./ particles.ρ .+ [apply_gravity()]) * dt
        @reset particles .= check_collusion.(particles)
        @reset particles.r .+= particles.v * dt

        rs = Iterators.product(x, y, z)
        rsv = reshape(collect(rs), length(x), length(z))
        color_field = color_smooth.(rsv, [particles], h)
        println("$n steps, $time s: $(size(color_field))")
        plot(
            scatter(getindex.(particles.r, 1), getindex.(particles.r, 3), clims=(0, 10), zcolor=norm.(particles.v), xlims=[0, 3], ylims=[-0.1, 1.4], title="particles, $time s", label="velocity", aspect_ratio=:equal),
            heatmap(x, z, color_field', aspect_ratio=:equal, clims=(0, 3.0), title="color field, $time s", xlims=[0, 3], ylims=[-0.1, 1.4]), layout=l
        )
        frame(anim)
    end
    gif(anim, "anim_fps1.gif", fps=5)

    println("done")
end

@time main()