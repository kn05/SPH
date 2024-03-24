using Pkg
Pkg.activate(".")

using Plots
using StaticArrays
using LinearAlgebra
using Accessors
using Zygote

function χ(p) # Indicator function
    if p
        return 1.0
    else
        return 0.0
    end
end

function W_poly6(r, h) # kernel function
    return 315 / (64 * pi * h^9) * (h^2 - r^2)^3 * χ(0 <= r <= h)
end

struct Particle
    m::Float64
    ρ::Float64 # density field
    p::Float64
    r::SVector{3,Float64}
    v::SVector{3,Float64} # velocity field
    f::SVector{3,Float64}
end

num_particles = 1000
h = 0.1
particles = Vector{Particle}(undef, num_particles)

for i in 1:num_particles
    m = rand()
    ρ = rand()
    p = rand()
    r = SVector(rand(), rand(), rand())
    v = SVector(rand(), rand(), rand())
    f = SVector(rand(), rand(), rand())

    particles[i] = Particle(m, ρ, p, r, v, f)
end

function calculate_p(particles, i)
    sum_p = 0
    for (j, p_j) in enumerate(particles)
        if j == i
            continue
        end
        sum_p += -p_j.m * (particles[i].p + p_j.p) / (2 * p_j.ρ) * gradient(r -> W_poly6(r, h), norm(particles[i].r - p_j.r))[1]
    end
    return sum_p
end

for (i, particle) in enumerate(particles)
    println(calculate_p(particles, i))
end

# 주어진 좌표 리스트
x = [coord[1] for coord in getfield.(particles, :r)]
y = [coord[2] for coord in getfield.(particles, :r)]
z = [coord[3] for coord in getfield.(particles, :r)]

# 3D 산점도 그리기


function C_S(r, particles)
    sum = 0
    for p in particles
        sum += p.m * (1 / p.ρ) * W_poly6(norm(r - p.r), h)
    end
    return sum
end

color_field = C_S.(getfield.(particles, :r), [particles])
color_field /= 200000

scatter3d(x, y, z, zcolor=color_field, alpha=0.7)