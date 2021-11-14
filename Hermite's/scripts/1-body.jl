# This script provides an implementation of
# the 4th-order Hermite's method for solving
# a system of differential equations and
# compares it to the exact solutions and
# other methods while solving a 1-body problem

println('\n', " "^4, "> Loading the packages...")

using LinearAlgebra
using Printf

# Define the value of ϰ
const ϰ = -1

# Integrate equations of motion using the
# 4th-order Runge-Kutta's method, return
# the values of position and velocity on
# the last step
function rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F<:AbstractFloat,I<:Unsigned}
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Compute the solution
    for _ = 1:n
        k₁ = v
        a₁ = ϰ * r / norm(r)^3
        k₂ = v + h * a₁ / 2
        a₂ = ϰ * (r + h * k₁ / 2) / norm(r + h * k₁ / 2)^3
        k₃ = v + h * a₂ / 2
        a₃ = ϰ * (r + h * k₂ / 2) / norm(r + h * k₂ / 2)^3
        k₄ = v + h * a₃
        a₄ = ϰ * (r + h * k₃) / norm(r + h * k₃)^3
        r += h / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
        v += h / 6 * (a₁ + 2 * a₂ + 2 * a₃ + a₄)
    end
    return r, v
end

# Integrate equations of motion using the
# 4th-order Hermite's method, return
# the values of position and velocity on
# the last step
function hermite(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F<:AbstractFloat,I<:Unsigned}
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Define the acceleration and jerk functions
    acc(r) = ϰ * r / norm(r)^3
    jerk(r, v) = ϰ * (v / norm(r)^3 - 3 * (r ⋅ v) .* r / norm(r)^5)
    # Compute the solution
    for _ = 1:n
        rₒ, vₒ = copy(r), copy(v)
        aₒ, jₒ = acc(r), jerk(r, v)

        r += v * h + aₒ * (h^2 / 2) + jₒ * (h^3 / 6)
        v += aₒ * h + jₒ * (h^2 / 2)

        a, j = acc(r), jerk(r, v)

        v = vₒ + (aₒ + a) * (h / 2) + (jₒ - j) * (h^2 / 12)
        r = rₒ + (vₒ + v) * (h / 2) + (aₒ - a) * (h^2 / 12)
    end
    return r, v
end

println(" "^4, "> Integrating the 1-body problem...")

# Define the initial values of the position and velocity
r₀ = [1.0, 0.0]
v₀ = [0.0, 0.5]

# Calculate integrals and some of the orbit parameters
E₀ = 1 / 2 * norm(v₀)^2 - ϰ^2 / norm(r₀)

# Print the initial values
println(
    '\n',
    " "^6, "r₀: ", r₀[1], " ", r₀[2], '\n',
    " "^6, "v₀: ", v₀[1], " ", v₀[2], '\n',
    " "^6, "E₀: ", E₀,
)

# Define the output directories
const data_dir = "$(@__DIR__)/../data/1-body"
const files = joinpath.(
    data_dir,
    ["rk4.dat", "hermite.dat"]
)

# Prepare a list of methods
const methods = (rk4, hermite)

# Integrate equations of motion using all available
# methods and print the values of position and velocity
# on the last step
function integrate(h::Vector{F}, n::Vector{I}) where {F<:AbstractFloat,I<:Unsigned}
    # Open the data files
    io = open.(files, "a")

    # Prepare results vectors
    r = [Vector{Vector{F}}() for _ = 1:length(io)]
    v = [Vector{Vector{F}}() for _ = 1:length(io)]
    E = [Vector{F}() for _ = 1:length(io)]
    ΔE = [Vector{F}() for _ = 1:length(io)]

    # For each pair of parameters
    for i in eachindex(h)
        # Integrate and get the results of the last steps
        for j in eachindex(io)
            rᵢ, vᵢ = methods[j](r₀, v₀, h[i], n[i])

            push!(r[j], rᵢ)
            push!(v[j], vᵢ)

            Eᵢ = 1 / 2 * norm(vᵢ)^2 - ϰ^2 / norm(rᵢ)
            ΔEᵢ = abs(E₀ - Eᵢ)

            push!(E[j], Eᵢ)
            push!(ΔE[j], ΔEᵢ)
        end
    end

    # Get the format string based on the index of the iteration
    function get_format(i)
        Printf.Format(
            "\$ 10^{-$(length("$(n[i])") - 1)} \$ " *
            "& \$ $(n[i] % 10 == 0 ? "10^$(length("$(n[i])") - 1)" : n[i]) \$ " *
            "& \$ %.14f \$ " *
            "& \$ %.14f \$ " *
            "\\\\\n"
        )
    end

    # Print the data in the files in the LaTeX format

    for j in eachindex(io)
        println(io[j], "# Positions")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), r[j][i][1], r[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Velocities")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), v[j][i][1], v[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Total energy")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), E[j][i], ΔE[j][i])
        end
        println(io[j])
        close(io[j])
    end
end

# Truncate the previous results
open.(files; truncate = true)

# Integrate n = 2^m iterations
integrate(
    [1e-2, 1e-3, 1e-4, 1e-5], #, 1e-6, 1e-7],
    UInt.([10^i for i = 2:5]),
)

# Integrate the full cycle
integrate(
    [1e-2, 1e-3, 1e-4, 1e-5], #, 1e-6, 1e-7],
    UInt.([271, 2714, 27141, 271408]), #, 2714081, 27140809 ]),
)

println()
