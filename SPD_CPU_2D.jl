using Pkg
Pkg.activate(".")

using Plots
using StaticArrays
using LinearAlgebra
using StructArrays
using ProgressMeter
using BenchmarkTools
using WriteVTK

Vec2{T} = SVector{2,T}

struct Particle{T}
    mass::T
    dens::T # density field
    pres::T
    pos::Vec2{T}
    vel::Vec2{T} # velocity field
    acc::Vec2{T}
    force::Vec2{T}
end

mutable struct Cell
    paticle_index::Vector{Int64}
end

function χ(p::Bool) # Indicator function
    if p
        return 1.0
    else
        return 0.0
    end
end

function W_poly6(pos::Vec2{T}, h::T) where {T} # kernel function
    r_norm = norm(pos)
    return 315 / (64 * pi * h^9) * (h^2 - r_norm^2)^3 * χ(0 <= r_norm <= h)
end

function ∇W_poly6(pos::Vec2{T}, h::T) where {T} # kernel function
    return 315 / (64 * pi * h^9) * -6 * (-h^2 + pos[1]^2 + pos[2]^2)^2 * pos
end

function ∇2_W_viscosity(pos::Vec2{T}, h::T) where {T}
    r_norm = norm(pos)
    45 / (pi * h^6) * (h - r_norm)
end

function apply_gravity()
    return SVector(0.0, -1.8)
end


function find_cell_index(point, x_range, y_range)
    if point[1] < first(x_range) || point[1] > last(x_range) || point[2] < first(y_range) || point[2] > last(y_range)
        return -1
    end
    xy_grid = collect.(Iterators.product(x_range, y_range))
    _, index = findmin(norm.(xy_grid .- [point]))
    return length(x_range) * (index[2] - 1) + index[1]
end

function main()
    k = 1000.0 # gas constant or stiffness
    ρ0 = 1000.0
    μ = 8.0  # viscosity constant
    kernel = 0.4
    x_length = 2.56
    y_length = 2.56
    cell_size = kernel
    x_range = 0:cell_size:x_length
    y_range = 0:cell_size:y_length
    total_cell = length(x_range) * length(y_range)

    dt = 0.05
    times = range(0, 5, step=dt)

    pos_init = [[x, y] for x in 0.3*x_length:0.6*kernel:0.7*x_length for y in 0.3*y_length:0.6*kernel:0.9*y_length]
    num_particles = length(pos_init)
    particles = begin
        mass = fill(0.02, num_particles)
        dens = fill(1.0, num_particles)
        pres = fill(4.9, num_particles)
        pos = pos_init
        vel = [[0.0, 0.0] for _ in 1:num_particles]
        acc = [[0.0, 0.0] for _ in 1:num_particles]
        force = [[0.0, 0.0] for _ in 1:num_particles]
        StructArray{Particle{Float64}}((mass, dens, pres, pos, vel, acc, force))
    end
    cells = SVector{total_cell,Vector{Int64}}(fill(Int64[], total_cell)...)

    l = @layout([a; b])
    anim = Animation()

    force_history = []
    vel_history = []

    @showprogress for (n, time) ∈ enumerate(times)
        # calc cell index 
        for (i, particle) in enumerate(particles)
            cell_index = find_cell_index(particle.pos, x_range, y_range)
            push!(cells[cell_index], i)
        end

        # calc density, pressure, and force
        for (i, p) in enumerate(particles)
            near = collect.(Iterators.product(-cell_size:cell_size:cell_size, -cell_size:cell_size:cell_size))
            cell_index = find_cell_index(p.pos, x_range, y_range)
            near_cell_indexs = find_cell_index.([p.pos] .+ near, [x_range], [y_range]) |> filter(x -> x != -1)
            p.dens = 0.0
            f_pressure = SVector(0.0, 0.0)
            f_viscosity = SVector(0.0, 0.0)
            f_gravity = SVector(0.0, 0.0)

            paticle_indexs = collect(Iterators.flatten(cells[near_cell_indexs]))
            for particle_index in paticle_indexs
                np = particles[particle_index]
                dist = norm(p.pos - np.pos)
                if dist < kernel
                    p.dens += p.mass * W_poly6(p.pos - np.pos, kernel)
                    p.pres = k * (p.dens - ρ0)
                    f_pressure += -np.mass * (p.pres + np.pres) / (2 * np.dens) * ∇W_poly6(p.pos - np.pos, kernel)
                    f_viscosity += np.mass * (np.vel - p.vel) / np.dens * ∇2_W_viscosity(p.pos - np.pos, kernel)
                    f_viscosity *= μ
                end
            end

            f_gravity = p.dens * apply_gravity()

            p.force = f_pressure + f_viscosity + f_gravity
            particles[i] = p
        end

        for (i, p) in enumerate(particles)
            p.acc = p.force / p.dens
            p.vel += p.acc * dt
            p.pos += p.vel * dt
            if p.pos[1] < first(x_range)
                p.vel[1] *= wallDamping
                p.pos[1] = first(x_range)
            end
            if p.pos[1] >= last(x_range)
                p.vel[1] *= wallDamping
                p.pos[1] = last(x_range) - 0.0001
            end
            if p.pos[2] < first(y_range)
                p.vel[2] *= wallDamping
                p.pos[2] = first(y_range)
            end
            if p.pos[2] >= last(y_range)
                p.vel[2] *= wallDamping
                p.pos[2] = last(y_range) - 0.0001
            end
            particles[i] = p
        end
        push!(force_history, norm(particles[1].force))
        push!(vel_history, norm(particles[1].vel))
        plot(scatter(getindex.(particles.pos, 1), getindex.(particles.pos, 2), zcolor=norm.(particles.vel), xlims=[0, x_length], ylims=[0, y_length], title="particles, $time s"),)
        frame(anim)
    end

    gif(anim, "anim.gif", fps=5)
    println("done")

    plot(force_history)

end

main()