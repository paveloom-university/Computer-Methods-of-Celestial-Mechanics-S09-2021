# This script provides implementations of
# multi-step methods of different orders
# for solving a system of differential equations
# and compares them to the exact solutions and
# other methods while solving a 1-body problem

println('\n', " "^4, "> Loading the packages...")

using LinearAlgebra
using Printf

# Define the value of ϰ
const ϰ = -1

# Integrate equations of motion using the
# Euler's method, return the values of position
# and velocity on the last step
function euler(
    r₀₁::Vector{F},
    r₀₂::Vector{F},
    r₀₃::Vector{F},
    v₀₁::Vector{F},
    v₀₂::Vector{F},
    v₀₃::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F},Vector{F},Vector{F},Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀₁)
    # Prepare the output vectors
    r₁ = copy(r₀₁)
    r₂ = copy(r₀₂)
    r₃ = copy(r₀₃)
    v₁ = copy(v₀₁)
    v₂ = copy(v₀₂)
    v₃ = copy(v₀₃)
    # Compute the solutions (in-place)
    for _ in 1:n
        ρ₁₂ = norm(r₁ - r₂)^3
        ρ₁₃ = norm(r₁ - r₃)^3
        ρ₂₃ = norm(r₂ - r₃)^3
        for k in 1:N
            a₁₂ = ϰ * (r₁[k] - r₂[k]) / ρ₁₂
            a₁₃ = ϰ * (r₁[k] - r₃[k]) / ρ₁₃
            a₂₃ = ϰ * (r₂[k] - r₃[k]) / ρ₂₃
            a₂₁ = -a₁₂
            a₃₁ = -a₁₃
            a₃₂ = -a₂₃
            r₁[k] += h * v₁[k]
            r₂[k] += h * v₂[k]
            r₃[k] += h * v₃[k]
            v₁[k] += h * (a₁₂ + a₁₃)
            v₂[k] += h * (a₂₁ + a₂₃)
            v₃[k] += h * (a₃₁ + a₃₂)
        end
    end
    return r₁, r₂, r₃, v₁, v₂, v₃
end

# Integrate the three-body problem using the
# two-step Adams–Bashforth's method, return the
# values of position and velocity on the last step
function ab2(
    r₀₁::Vector{F},
    r₀₂::Vector{F},
    r₀₃::Vector{F},
    v₀₁::Vector{F},
    v₀₂::Vector{F},
    v₀₃::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F},Vector{F},Vector{F},Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀₁)
    # Prepare the output vectors
    r₁ = copy(r₀₁)
    r₂ = copy(r₀₂)
    r₃ = copy(r₀₃)
    v₁ = copy(v₀₁)
    v₂ = copy(v₀₂)
    v₃ = copy(v₀₃)
    # Prepare buffers for previous values
    r₁_ₖ₋₁ = copy(r₀₁)
    r₂_ₖ₋₁ = copy(r₀₂)
    r₃_ₖ₋₁ = copy(r₀₃)
    v₁_ₖ₋₁ = copy(v₀₁)
    v₂_ₖ₋₁ = copy(v₀₂)
    v₃_ₖ₋₁ = copy(v₀₃)
    # Compute the second value of the solution
    # by using the one-step Euler's method
    r₁, r₂, r₃, v₁, v₂, v₃ = euler(r₀₁, r₀₂, r₀₃, v₀₁, v₀₂, v₀₃, h, UInt(1))
    # Define a couple of independent coefficients
    k₁ = 3 / 2 * h
    k₂ = -1 / 2 * h
    # Compute the rest in two steps
    for _ in 2:n
        ρ₁₂_ₖ = norm(r₁ - r₂)^3
        ρ₁₃_ₖ = norm(r₁ - r₃)^3
        ρ₂₃_ₖ = norm(r₂ - r₃)^3
        ρ₁₂_ₖ₋₁ = norm(r₁_ₖ₋₁ - r₂_ₖ₋₁)^3
        ρ₁₃_ₖ₋₁ = norm(r₁_ₖ₋₁ - r₃_ₖ₋₁)^3
        ρ₂₃_ₖ₋₁ = norm(r₂_ₖ₋₁ - r₃_ₖ₋₁)^3
        for k in 1:N
            a₁₂_ₖ = k₁ * ϰ * (r₁[k] - r₂[k]) / ρ₁₂_ₖ
            a₁₃_ₖ = k₁ * ϰ * (r₁[k] - r₃[k]) / ρ₁₃_ₖ
            a₂₃_ₖ = k₁ * ϰ * (r₂[k] - r₃[k]) / ρ₂₃_ₖ
            a₁₂_ₖ₋₁ = k₂ * ϰ * (r₁_ₖ₋₁[k] - r₂_ₖ₋₁[k]) / ρ₁₂_ₖ₋₁
            a₁₃_ₖ₋₁ = k₂ * ϰ * (r₁_ₖ₋₁[k] - r₃_ₖ₋₁[k]) / ρ₁₃_ₖ₋₁
            a₂₃_ₖ₋₁ = k₂ * ϰ * (r₂_ₖ₋₁[k] - r₃_ₖ₋₁[k]) / ρ₂₃_ₖ₋₁

            a₂₁_ₖ = -a₁₂_ₖ
            a₃₁_ₖ = -a₁₃_ₖ
            a₃₂_ₖ = -a₂₃_ₖ
            a₂₁_ₖ₋₁ = -a₁₂_ₖ₋₁
            a₃₁_ₖ₋₁ = -a₁₃_ₖ₋₁
            a₃₂_ₖ₋₁ = -a₂₃_ₖ₋₁

            r₁_ₖ₋₁[k] = r₁[k]
            r₂_ₖ₋₁[k] = r₂[k]
            r₃_ₖ₋₁[k] = r₃[k]

            r₁[k] += k₁ * v₁[k] + k₂ * v₁_ₖ₋₁[k]
            r₂[k] += k₁ * v₂[k] + k₂ * v₂_ₖ₋₁[k]
            r₃[k] += k₁ * v₃[k] + k₂ * v₃_ₖ₋₁[k]

            v₁_ₖ₋₁[k] = v₁[k]
            v₂_ₖ₋₁[k] = v₂[k]
            v₃_ₖ₋₁[k] = v₃[k]

            v₁[k] += a₁₂_ₖ + a₁₃_ₖ + a₁₂_ₖ₋₁ + a₁₃_ₖ₋₁
            v₂[k] += a₂₁_ₖ + a₂₃_ₖ + a₂₁_ₖ₋₁ + a₂₃_ₖ₋₁
            v₃[k] += a₃₁_ₖ + a₃₂_ₖ + a₃₁_ₖ₋₁ + a₃₂_ₖ₋₁
        end
    end
    return r₁, r₂, r₃, v₁, v₂, v₃
end

# Integrate the three-body problem using the
# three-step Adams–Bashforth's method, return the
# values of position and velocity on the last step
function ab3(
    r₀₁::Vector{F},
    r₀₂::Vector{F},
    r₀₃::Vector{F},
    v₀₁::Vector{F},
    v₀₂::Vector{F},
    v₀₃::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F},Vector{F},Vector{F},Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀₁)
    # Prepare the output vectors
    r₁ = copy(r₀₁)
    r₂ = copy(r₀₂)
    r₃ = copy(r₀₃)
    v₁ = copy(v₀₁)
    v₂ = copy(v₀₂)
    v₃ = copy(v₀₃)
    # Prepare buffers for previous values
    r₁_ₖ₋₁ = copy(r₀₁)
    r₂_ₖ₋₁ = copy(r₀₂)
    r₃_ₖ₋₁ = copy(r₀₃)
    v₁_ₖ₋₁ = copy(v₀₁)
    v₂_ₖ₋₁ = copy(v₀₂)
    v₃_ₖ₋₁ = copy(v₀₃)
    r₁_ₖ₋₂ = copy(r₀₁)
    r₂_ₖ₋₂ = copy(r₀₂)
    r₃_ₖ₋₂ = copy(r₀₃)
    v₁_ₖ₋₂ = copy(v₀₁)
    v₂_ₖ₋₂ = copy(v₀₂)
    v₃_ₖ₋₂ = copy(v₀₃)
    # Compute the second value of the solution
    # by using the one-step Euler's method
    r₁, r₂, r₃, v₁, v₂, v₃ = euler(r₀₁, r₀₂, r₀₃, v₀₁, v₀₂, v₀₃, h, UInt(1))
    # Compute the third value of the solution
    # by using the two-step Adams–Bashforth's method
    r₁_ₖ₋₁, r₂_ₖ₋₁, r₃_ₖ₋₁, v₁_ₖ₋₁, v₂_ₖ₋₁, v₃_ₖ₋₁ = r₁, r₂, r₃, v₁, v₂, v₃
    r₁, r₂, r₃, v₁, v₂, v₃ = ab2(r₀₁, r₀₂, r₀₃, v₀₁, v₀₂, v₀₃, h, UInt(2))
    # Define a couple of independent coefficients
    k₁ = 23 / 12 * h
    k₂ = -4 / 3 * h
    k₃ = 5 / 12 * h
    # Compute the rest in two steps
    for _ in 3:n
        ρ₁₂_ₖ = norm(r₁ - r₂)^3
        ρ₁₃_ₖ = norm(r₁ - r₃)^3
        ρ₂₃_ₖ = norm(r₂ - r₃)^3
        ρ₁₂_ₖ₋₁ = norm(r₁_ₖ₋₁ - r₂_ₖ₋₁)^3
        ρ₁₃_ₖ₋₁ = norm(r₁_ₖ₋₁ - r₃_ₖ₋₁)^3
        ρ₂₃_ₖ₋₁ = norm(r₂_ₖ₋₁ - r₃_ₖ₋₁)^3
        ρ₁₂_ₖ₋₂ = norm(r₁_ₖ₋₂ - r₂_ₖ₋₂)^3
        ρ₁₃_ₖ₋₂ = norm(r₁_ₖ₋₂ - r₃_ₖ₋₂)^3
        ρ₂₃_ₖ₋₂ = norm(r₂_ₖ₋₂ - r₃_ₖ₋₂)^3
        for k in 1:N
            a₁₂_ₖ = k₁ * ϰ * (r₁[k] - r₂[k]) / ρ₁₂_ₖ
            a₁₃_ₖ = k₁ * ϰ * (r₁[k] - r₃[k]) / ρ₁₃_ₖ
            a₂₃_ₖ = k₁ * ϰ * (r₂[k] - r₃[k]) / ρ₂₃_ₖ
            a₁₂_ₖ₋₁ = k₂ * ϰ * (r₁_ₖ₋₁[k] - r₂_ₖ₋₁[k]) / ρ₁₂_ₖ₋₁
            a₁₃_ₖ₋₁ = k₂ * ϰ * (r₁_ₖ₋₁[k] - r₃_ₖ₋₁[k]) / ρ₁₃_ₖ₋₁
            a₂₃_ₖ₋₁ = k₂ * ϰ * (r₂_ₖ₋₁[k] - r₃_ₖ₋₁[k]) / ρ₂₃_ₖ₋₁
            a₁₂_ₖ₋₂ = k₃ * ϰ * (r₁_ₖ₋₂[k] - r₂_ₖ₋₂[k]) / ρ₁₂_ₖ₋₂
            a₁₃_ₖ₋₂ = k₃ * ϰ * (r₁_ₖ₋₂[k] - r₃_ₖ₋₂[k]) / ρ₁₃_ₖ₋₂
            a₂₃_ₖ₋₂ = k₃ * ϰ * (r₂_ₖ₋₂[k] - r₃_ₖ₋₂[k]) / ρ₂₃_ₖ₋₂

            a₂₁_ₖ = -a₁₂_ₖ
            a₃₁_ₖ = -a₁₃_ₖ
            a₃₂_ₖ = -a₂₃_ₖ
            a₂₁_ₖ₋₁ = -a₁₂_ₖ₋₁
            a₃₁_ₖ₋₁ = -a₁₃_ₖ₋₁
            a₃₂_ₖ₋₁ = -a₂₃_ₖ₋₁
            a₂₁_ₖ₋₂ = -a₁₂_ₖ₋₂
            a₃₁_ₖ₋₂ = -a₁₃_ₖ₋₂
            a₃₂_ₖ₋₂ = -a₂₃_ₖ₋₂

            r₁_ₖ₋₂[k] = r₁_ₖ₋₁[k]
            r₂_ₖ₋₂[k] = r₂_ₖ₋₁[k]
            r₃_ₖ₋₂[k] = r₃_ₖ₋₁[k]
            r₁_ₖ₋₁[k] = r₁[k]
            r₂_ₖ₋₁[k] = r₂[k]
            r₃_ₖ₋₁[k] = r₃[k]

            r₁[k] += k₁ * v₁[k] + k₂ * v₁_ₖ₋₁[k] + k₃ * v₁_ₖ₋₂[k]
            r₂[k] += k₁ * v₂[k] + k₂ * v₂_ₖ₋₁[k] + k₃ * v₂_ₖ₋₂[k]
            r₃[k] += k₁ * v₃[k] + k₂ * v₃_ₖ₋₁[k] + k₃ * v₃_ₖ₋₂[k]

            v₁_ₖ₋₂[k] = v₁_ₖ₋₁[k]
            v₂_ₖ₋₂[k] = v₂_ₖ₋₁[k]
            v₃_ₖ₋₂[k] = v₃_ₖ₋₁[k]
            v₁_ₖ₋₁[k] = v₁[k]
            v₂_ₖ₋₁[k] = v₂[k]
            v₃_ₖ₋₁[k] = v₃[k]

            v₁[k] += a₁₂_ₖ + a₁₃_ₖ + a₁₂_ₖ₋₁ + a₁₃_ₖ₋₁ + a₁₂_ₖ₋₂ + a₁₃_ₖ₋₂
            v₂[k] += a₂₁_ₖ + a₂₃_ₖ + a₂₁_ₖ₋₁ + a₂₃_ₖ₋₁ + a₂₁_ₖ₋₂ + a₂₃_ₖ₋₂
            v₃[k] += a₃₁_ₖ + a₃₂_ₖ + a₃₁_ₖ₋₁ + a₃₂_ₖ₋₁ + a₃₁_ₖ₋₂ + a₃₂_ₖ₋₂
        end
    end
    return r₁, r₂, r₃, v₁, v₂, v₃
end

println(" "^4, "> Integrating the 3-body problem...")

# Define the initial values of the position and velocity
r₀₁ = [0.97, -0.2431]
r₀₂ = [-0.97, 0.2431]
r₀₃ = [0.0, 0.0]
v₀₁ = [0.4662, 0.4324]
v₀₂ = [0.4662, 0.4324]
v₀₃ = [-0.9324, -0.8647]

# Calculate total energy
E₀ = 1 / 2 * (norm(v₀₁)^2 + norm(v₀₂)^2 + norm(v₀₃)^2) - ϰ^2 *
    (1 / norm(r₀₁ - r₀₂) + 1 / norm(r₀₃ - r₀₂) + 1 / norm(r₀₃ - r₀₁))

# Print the initial values
println(
    '\n',
    " "^6, "r₀₁: ", r₀₁[1], " ", r₀₁[2], '\n',
    " "^6, "r₀₂: ", r₀₂[1], " ", r₀₂[2], '\n',
    " "^6, "r₀₃: ", r₀₃[1], " ", r₀₃[2], '\n',
    " "^6, "v₀₁: ", v₀₁[1], " ", v₀₁[2], '\n',
    " "^6, "v₀₂: ", v₀₂[1], " ", v₀₂[2], '\n',
    " "^6, "v₀₃: ", v₀₃[1], " ", v₀₃[2], '\n',
    " "^6, "E₀: ", E₀,
)

# Define the output directories
const data_dir = "$(@__DIR__)/../data/3-body"
const files = joinpath.(
    data_dir,
    [
        "euler.dat",
        "ab2.dat",
        "ab3.dat",
    ]
)

# Prepare a list of methods
const methods = (euler, ab2, ab3)

# Integrate equations of motion using all available
# methods and write the values of position and velocity
# on the last step
function integrate(h::Vector{F}, n::Vector{I}) where {F <: AbstractFloat,I <: Unsigned}
    # Open the data files
    io = open.(files, "a")

    # Prepare results vectors

    r₁ = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    r₂ = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    r₃ = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    v₁ = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    v₂ = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    v₃ = [ Vector{Vector{F}}() for _ in 1:length(io) ]

    E = [ Vector{F}() for _ in 1:length(io) ]
    ΔE = [ Vector{F}() for _ in 1:length(io) ]

    # For each pair of parameters
    for i in eachindex(h)
        # Integrate and get the results of the last steps
        for j in eachindex(io)
            rᵢ₁, rᵢ₂, rᵢ₃, vᵢ₁, vᵢ₂, vᵢ₃ = methods[j](r₀₁, r₀₂, r₀₃, v₀₁, v₀₂, v₀₃, h[i], n[i])

            push!(r₁[j], rᵢ₁)
            push!(r₂[j], rᵢ₂)
            push!(r₃[j], rᵢ₃)
            push!(v₁[j], vᵢ₁)
            push!(v₂[j], vᵢ₂)
            push!(v₃[j], vᵢ₃)

            Eᵢ = 1 / 2 * (norm(vᵢ₁)^2 + norm(vᵢ₂)^2 + norm(vᵢ₃)^2) - ϰ^2 *
                (1 / norm(rᵢ₁ - rᵢ₂) + 1 / norm(rᵢ₃ - rᵢ₂) + 1 / norm(rᵢ₃ - rᵢ₁))
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
        println(io[j], "# Positions #1")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), r₁[j][i][1], r₁[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Positions #2")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), r₂[j][i][1], r₂[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Positions #3")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), r₃[j][i][1], r₃[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Velocities #1")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i),  v₁[j][i][1], v₁[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Velocities #2")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i),  v₂[j][i][1], v₂[j][i][2])
        end
    end

    for j in eachindex(io)
        println(io[j], "\n# Velocities #3")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i),  v₃[j][i][1], v₃[j][i][2])
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
open.(files; truncate=true)

# Integrate n = 2^m iterations
integrate(
    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    UInt.([ 10^i for i in 2:7 ]),
)

println()
