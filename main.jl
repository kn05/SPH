using Pkg
Pkg.activate(".")

using Plots
using StaticArrays
using LinearAlgebra
using Accessors
using Zygote
using StructArrays
using ProgressMeter

k = 1 # gas constant
μ = 1 # viscosity constant


struct Particle{T}
    m::T
    ρ::T # density field
    p::T
    r::SVector{3,T}
    v::SVector{3,T} # velocity field
    f::SVector{3,T}
end

function χ(p) # Indicator function
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
    45/(pi*h^6)*(h-r_norm)
end

function C_smooth(r, particles) # smoothed color field 
    sum = 0
    for p in particles
        sum += p.m * (1 / p.ρ) * W_poly6(norm(r - p.r), h)
    end
    return sum
end

function eval_pressure_force(particles, i)
    f_pressure = SVector(0, 0, 0)

    for (j, particle_j) in enumerate(particles)
        if i == j
            continue
        end
        f_pressure = -particle_j.m * (particles[i].p + particle_j.p) / (2 * particle_j.ρ) * gradient(r -> W_poly6(r, h), particles[i].r - particle_j.r)[1]
    end

    return f_pressure
end

function eval_viscosity_force(particles, i)
    f_viscosity = SVector(0, 0, 0)
    for (j, _) in enumerate(particles)
        if i == j
            continue
        end
        f_viscosity += particle.m[j] *(particle.v[j]-particle.v[i])/particle.ρ[j]*∇2_W_viscosity(particle.r[i]-particle.r[j], h)
    end
    f_viscosity *= μ
    return f_viscosity
end

# init 
grid = 10
num_particles = grid^3
h = 1 / grid 

particles = begin
    m = fill(1 / num_particles, num_particles)
    ρ = fill(1.0, num_particles)
    p = fill(1.0, num_particles)
    r = [SVector(rand(), rand(), rand()) for _ in 1:num_particles]
    v = [SVector(rand(), rand(), rand()) for _ in 1:num_particles]
    f = [SVector(0.0, 0.0, 0.0) for _ in 1:num_particles]

    StructArray{Particle}((m, ρ, p, r, v, f))
end

# calc pressure grad. force
@showprogress for (i, particle) in enumerate(particles)
    @reset particles.f[i] = calculate_p(particles, i)
end

begin
    x = [coord[1] for coord in particles.r]
    y = [coord[2] for coord in getfield.(particles, :r)]
    z = [coord[3] for coord in getfield.(particles, :r)]
    fx = [coord[1] for coord in getfield.(particles, :f)]
    fy = [coord[2] for coord in getfield.(particles, :f)]
    fz = [coord[3] for coord in getfield.(particles, :f)]
    scatter3d(x, y, z, xlims=[0, 1], ylims=[0, 1], zlims=[0, 1])
    quiver!(x, y, z, quiver=(x + fx, y + fy, z + fz), xlims=[0, 1], ylims=[0, 1], zlims=[0, 1])
    getfield.(particles, :f)
end



begin
    rs = [[x, y, 0.5] for x = -0.1:0.01:1.1, y = -0.1:0.01:1.1]
    flat_rs = vec(rs)
    color_field = C_S.(flat_rs, [particles])
    scatter(getindex.(flat_rs, 1), getindex.(flat_rs, 2), zcolor=color_field, alpha=1)
end


gradient(r -> W_poly6(r, h), 0.2)`