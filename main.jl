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
    n::SVector{3,T}
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

Base.:-(x::Tuple{Float64,Float64}, y::Tuple{Float64,Float64}) = x .- y
Base.:-(x::Tuple{Float64,Float64}, y::Vector{Float64}) = Tuple(x .- y)
function findnearest(p, x_range, y_range)
    xy_grid = collect(Iterators.product(x_range, y_range))
    idx = findmin(norm.(xy_grid .- [p]))[2]
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


function ∇(m::Matrix, h) #=h is size of cell=#
    x_length, y_length = size(m)
    diff_x = diff(m, dims=1)
    diff_y = diff(m, dims=2)
    dfdx = vcat(diff_x[1, :]', (diff_x[1:end-1, :] + diff_x[2:end, :]) / 2, diff_x[end, :]') / h
    dfdy = hcat(diff_y[:, 1], (diff_y[:, 1:end-1] + diff_y[:, 2:end]) / 2, diff_y[:, end]) / h
    new_mat = map(x -> SVector(x[1], 0.0, x[2]), zip(dfdx, dfdy))
end

function color_smooth(r::SVector{3,T}, particles, h) where {T}# smoothed color field 
    sum = 0.0
    for p in particles
        sum += p.m * (1 / p.ρ) * W_poly6(norm(r - p.r), h)
    end
    return sum
end

function eval_n(r, particles, h)
    n = gradient(r -> color_smooth(r, particles, h), r)[1]
    return n
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
    return 1.8 * [0.0, 0.0, -1.0]
end

function check_collusion(particle)
    if particle.r[1] < 0
        @reset particle.v[1] = -particle.v[1]
    end
    if particle.r[3] < 0
        @reset particle.v[3] = -particle.v[3]
    end
    if norm(particle.v) > 150
        @reset particle.v = SVector(0.0, 0.0, 0.0)
    end

    return particle
end

# init 
function main()
    k = 1.0 # gas constant
    μ = 8.0 # viscosity constant
    g = 0
    h = 0.1 # 0.04
    x_length = 2.56
    z_length = 2.56
    x_range = x_length*0.3:h*0.6:x_length*0.7
    z_range = 0:h*0.6:z_length*0.4
    num_particles = length(x_range) * length(z_range)

    particles = begin
        m = fill(0.02, num_particles)
        ρ = fill(1000.0, num_particles)
        p = fill(4.9, num_particles)
        r = [SVector(x + h / 10 * rand(), 0, z + h / 10 * rand()) for x in x_range for z in z_range]
        v = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]
        f = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]
        n = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]
        StructArray{Particle{Float64}}((m, ρ, p, r, v, f, n))
    end

    dt = 0.01
    x, y, z = 0:0.04:2.56, 0.0, 0:0.04:2.56
    times = range(0, 2, step=dt)

    pressure = zeros(num_particles)
    viscosity = zeros(num_particles)

    l = @layout([a; b])
    anim = Animation()

    @showprogress for (n, time) ∈ enumerate(times)
        @reset particles.ρ = ρ_smooth.(particles.r, [particles], h)


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
        n_field = ∇(color_field, 0.01)
        particle_idxs = findnearest.(getindex.(particles.r, [[1, 3]]), [x_range], [z_range])
        @reset particles.n = getindex.([n_field], particle_idxs)

        plot(
            scatter(getindex.(particles.r, 1), getindex.(particles.r, 3), clims=(0, 25), zcolor=norm.(particles.n), xlims=[0, x_length], ylims=[0, x_length], title="particles, $time s", label="n", aspect_ratio=:equal, markersize=0.5),
            heatmap(x, z, norm.(n_field)', aspect_ratio=:equal, clims=(0, 25), title="n, $time s", xlims=[0, x_length], ylims=[-0.1, x_length]), layout=l
        )
        frame(anim)
    end
    gif(anim, "수치조정버전2_anim_fps5.gif", fps=5)
    println("done")
end

@time main()