# This script provides implementations of
# multi-step methods of different orders
# for assessing their performance on a test
# equation y' = λ y

println('\n', " "^4, "> Loading the packages...")

using Printf

# Define the value of λ
const λ = -1.5

# Integrate the test equation using the
# Euler's method, return the value of the
# solution at the last step
function euler(
    y₀::F,
    h::F,
    n::I,
)::F where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare a buffer for the solution
    y = y₀
    # Compute the solution
    for _ in 1:n
        y += h * λ * y
    end
    return y
end

# Integrate the test equation using the
# two-step Adams–Bashforth's method, return
# the value of the solution at the last step
function ab2(
    y₀::F,
    h::F,
    n::I,
)::F where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare a buffer for the solution
    y = y₀
    # Prepare a buffer for previous values
    yₖ₋₁ = y₀
    # Compute the second value of the solution
    # using the one-step Euler's method
    y = euler(y₀, h, UInt(1))
    # Compute the rest in two steps
    for _ in 2:n
        k = y
        y += h * λ * (3 / 2 * y - 1 / 2 * yₖ₋₁)
        yₖ₋₁ = k
    end
    return y
end


# Integrate the test equation using the
# 4th order Runge-Kutta's method, return
# the value of the solution at the last step
function rk4(
    y₀::F,
    h::F,
    n::I,
)::F where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare a buffer for the solution
    y = y₀
    # Compute the rest in two steps
    for _ in 1:n
        k₁ = λ * y
        k₂ = λ * (y + h * k₁ / 2)
        k₃ = λ * (y + h * k₂ / 2)
        k₄ = λ * (y + h * k₃)
        y += h / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    end
    return y
end

# Integrate the test equation using the
# three-step Adams–Bashforth's method
# (with Euler's method as the starter),
# return the value of the solution at
# the last step
function ab3_euler(
    y₀::F,
    h::F,
    n::I,
)::F where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare a buffer for the solution
    y = y₀
    # Prepare buffers for previous values
    yₖ₋₁ = y₀
    yₖ₋₂ = y₀
    # Compute the second value of the solution
    # using the one-step Euler's method
    y = euler(y₀, h, UInt(1))
    # Compute the third value of the solution
    # using the two-step Adams–Bashforth's method
    yₖ₋₁ = y
    y = ab2(y₀, h, UInt(2))
    # Compute the rest in two steps
    for _ in 3:n
        k = y
        y += h * λ * (23 / 12 * y - 4 / 3 * yₖ₋₁ + 5 / 12 * yₖ₋₂)
        yₖ₋₂ = yₖ₋₁
        yₖ₋₁ = k
    end
    return y
end

# Integrate the test equation using the
# three-step Adams–Bashforth's method
# (with 4th order Runge-Kutta's method as the
# starter), return the value of the solution
# at the last step
function ab3_rk4(
    y₀::F,
    h::F,
    n::I,
)::F where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare a buffer for the solution
    y = y₀
    # Prepare buffers for previous values
    yₖ₋₁ = y₀
    yₖ₋₂ = y₀
    # Compute the second and third values of the solution
    # using the 4th-order Runge-Kutta's method
    yₖ₋₁ = rk4(y₀, h, UInt(1))
    y = rk4(y₀, h, UInt(2))
    # Compute the rest in two steps
    for _ in 3:n
        k = y
        y += h * λ * (23 / 12 * y - 4 / 3 * yₖ₋₁ + 5 / 12 * yₖ₋₂)
        yₖ₋₂ = yₖ₋₁
        yₖ₋₁ = k
    end
    return y
end

println(" "^4, "> Integrating the test equation...")

# Define the initial values of the position and velocity
y₀ = 1.0

# Calculate the analytical solution
yₐ = ℯ^λ

# Print the initial values
println(
    '\n',
    " "^6, "y₀: ", y₀,
)

# Define the output directories
const data_dir = "$(@__DIR__)/../data/test"
const files = joinpath.(
    data_dir,
    [
        "euler.dat",
        "ab2.dat",
        "rk4.dat",
        "ab3_euler.dat",
        "ab3_rk4.dat",
    ]
)

# Prepare a list of methods
const methods = (euler, ab2, rk4, ab3_euler, ab3_rk4)

# Integrate the test equation using all available
# methods and write the values of the solutions
# on the last step
function integrate(h::Vector{F}, n::Vector{I}) where {F <: AbstractFloat,I <: Unsigned}
    # Open the data files
    io = open.(files, "a")

    # Prepare results vectors
    y = [ Vector{F}() for _ in 1:length(io) ]
    Δy = [ Vector{F}() for _ in 1:length(io) ]

    # For each pair of parameters
    for i in eachindex(h)
        # Integrate and get the results of the last steps
        for j in eachindex(io)
            yᵢ = methods[j](y₀, h[i], n[i])
            Δyᵢ = abs(yₐ - yᵢ)
            push!(y[j], yᵢ)
            push!(Δy[j], Δyᵢ)
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
        println(io[j], "# Solution")
        for i in eachindex(h)
            Printf.format(io[j], get_format(i), y[j][i], Δy[j][i])
        end
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
