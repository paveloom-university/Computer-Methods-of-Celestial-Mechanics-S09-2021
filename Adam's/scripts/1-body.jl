# This script provides implementations of
# multi-step methods of different orders
# for solving a system of differential equations
# and compares them to the exact solutions and
# other methods while solving a 1-body problem

println('\n', " "^4, "> Loading the packages...")

using LinearAlgebra
using NLsolve
using Printf

# Define the value of ϰ
const ϰ = -1

# Integrate equations of motion using the
# Euler's method, return the values of position
# and velocity on the last step
function euler(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Compute the solution
    for _ in 1:n
        a = ϰ * r / norm(r)^3
        r += h * v
        v += h * a
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Bashforth's method, return
# the values of position and velocity on the
# last step
function _ab2(
    r₀::Vector{F},
    v₀::Vector{F},
    r::Vector{F},
    v::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Define a couple of independent coefficients
    k₁ = 3 / 2 * h
    k₂ = -1 / 2 * h
    # Compute the rest in two steps
    for _ in 2:n
        a₁ = k₁ * ϰ * r / norm(r)^3
        a₂ = k₂ * ϰ * r₀ / norm(r₀)^3
        r₀ = r
        r += k₁ * v + k₂ * v₀
        v₀ = v
        v += a₁ + a₂
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Bashforth's method (with
# the Euler method as a starter), return
# the values of position and velocity on
# the last step
function ab2(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the one-step Euler's method
    r, v = euler(r₀, v₀, h, UInt(1))
    # Integrate and return results
    return _ab2(r₀, v₀, r, v, h, n)
end

# Integrate equations of motion using the
# two-step Adams–Bashforth's method (with
# the 4th-order Runge-Kutta method as a starter),
# return the values of position and velocity on
# the last step
function ab2_rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the 4th-order Runge-Kutta method
    r, v = rk4(r₀, v₀, h, UInt(1))
    # Integrate and return results
    return _ab2(r₀, v₀, r, v, h, n)
end

# Integrate equations of motion using the
# three-step Adams–Bashforth's method, return
# the values of position and velocity on the
# last step
function _ab3(
    r₀::Vector{F},
    v₀::Vector{F},
    r₁::Vector{F},
    v₁::Vector{F},
    r::Vector{F},
    v::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Define independent coefficients
    k₁ = 23 / 12 * h
    k₂ = -4 / 3 * h
    k₃ = 5 / 12 * h
    # Compute the rest in three steps
    for _ in 3:n
        a₁ = k₁ * ϰ * r / norm(r)^3
        a₂ = k₂ * ϰ * r₁ / norm(r₁)^3
        a₃ = k₃ * ϰ * r₀ / norm(r₀)^3
        r₀, r₁ = r₁, r
        r += k₁ * v + k₂ * v₁ + k₃ * v₀
        v₀, v₁ = v₁, v
        v += a₁ + a₂ + a₃
    end
    return r, v
end

# Integrate equations of motion using the
# three-step Adams–Bashforth's method (with
# the Euler method and the two-step
# Adams–Bashforth's method as starters),
# return the values of position and velocity
# on the last step
function ab3(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the one-step Euler's method
    r₁, v₁ = euler(r₀, v₀, h, UInt(1))
    # Compute the third value of the solution
    # using the two-step Adams–Bashforth's method
    r, v = ab2(r₀, v₀, h, UInt(2))
    return _ab3(r₀, v₀, r₁, v₁, r, v, h, n)
end

# Integrate equations of motion using the
# three-step Adams–Bashforth's method (with
# the 4th-order Runge-Kutta method as a starter),
# return the values of position and velocity
# on the last step
function ab3_rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second and third values of the
    # solution using the 4th-order Runge-Kutta method
    r₁, v₁ = rk4(r₀, v₀, h, UInt(1))
    r, v = rk4(r₀, v₀, h, UInt(2))
    return _ab3(r₀, v₀, r₁, v₁, r, v, h, n)
end

# Integrate equations of motion using the
# four-step Adams–Bashforth's method, return
# the values of position and velocity on the
# last step
function _ab4(
    r₀::Vector{F},
    v₀::Vector{F},
    r₁::Vector{F},
    v₁::Vector{F},
    r₂::Vector{F},
    v₂::Vector{F},
    r::Vector{F},
    v::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Define independent coefficients
    k₁ = 55 / 24 * h
    k₂ = -59 / 24 * h
    k₃ = 37 / 24 * h
    k₄ = -9 / 24 * h
    # Compute the rest in three steps
    for _ in 3:n
        a₁ = k₁ * ϰ * r / norm(r)^3
        a₂ = k₂ * ϰ * r₂ / norm(r₂)^3
        a₃ = k₃ * ϰ * r₁ / norm(r₁)^3
        a₄ = k₄ * ϰ * r₀ / norm(r₀)^3
        r₀, r₁, r₂ = r₁, r₂, r
        r += k₁ * v + k₂ * v₂ + k₃ * v₁ + k₄ * v₀
        v₀, v₁, v₂ = v₁, v₂, v
        v += a₁ + a₂ + a₃ + a₄
    end
    return r, v
end

# Integrate equations of motion using the
# four-step Adams–Bashforth's method (with
# the Euler method, two-step and three-step
# Adams–Bashforth's methods as starters),
# return the values of position and velocity
# on the last step
function ab4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the one-step Euler's method
    r₁, v₁ = euler(r₀, v₀, h, UInt(1))
    # Compute the third value of the solution
    # using the two-step Adams–Bashforth's method
    r₂, v₂ = ab2(r₀, v₀, h, UInt(2))
    # Compute the third value of the solution
    # using the three-step Adams–Bashforth's method
    r, v = ab3(r₀, v₀, h, UInt(3))
    return _ab4(r₀, v₀, r₁, v₁, r₂, v₂, r, v, h, n)
end

# Integrate equations of motion using the
# four-step Adams–Bashforth's method (with
# the 4th-order Runge-Kutta method as a starter),
# return the values of position and velocity
# on the last step
function ab4_rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second, third, and fourth values of
    # the solution using the 4th-order Runge-Kutta method
    r₁, v₁ = rk4(r₀, v₀, h, UInt(1))
    r₂, v₂ = rk4(r₀, v₀, h, UInt(2))
    r, v = rk4(r₀, v₀, h, UInt(3))
    return _ab4(r₀, v₀, r₁, v₁, r₂, v₂, r, v, h, n)
end

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
{F <: AbstractFloat,I <: Unsigned}
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Compute the solution
    for _ in 1:n
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
# backward Euler's method, return the values
# of position and velocity on the last step
function b_euler(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Compute the solution
    for _ in 1:n
        function f!(F, x)
            F[1:N] = -x[1:N] + r + h * x[N+1:end]
            F[N+1:end] = -x[N+1:end] + v + h * ϰ * x[1:N] / norm(x[1:N])^3
        end
        res = nlsolve(f!, [r; v], autodiff=:forward, method=:newton, ftol=1e-16).zero
        r, v = res[1:N], res[N+1:end]
    end
    return r, v
end

# Integrate equations of motion using the
# trapezoidal rule, return the values
# of position and velocity on the last step
function trapezoidal(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare the output vectors
    r = copy(r₀)
    v = copy(v₀)
    # Compute the solution
    for _ in 1:n
        ρₖ = norm(r)^3
        function f!(F, x)
            F[1:N] = -x[1:N] + r + h / 2 * (x[N+1:end] + v)
            F[N+1:end] = -x[N+1:end] + v + h / 2 * ϰ * (x[1:N] / norm([x[1:N]])^3 + r / ρₖ)
        end
        res = nlsolve(f!, [r; v], autodiff=:forward, method=:newton, ftol=1e-16).zero
        r, v = res[1:N], res[N+1:end]
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Moulton's method, return
# the values of position and velocity on the
# last step
function _am2(
    r₀::Vector{F},
    v₀::Vector{F},
    r::Vector{F},
    v::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Define independent coefficients
    k₁ = 5 / 12 * h
    k₂ = 2 / 3 * h
    k₃ = -1 / 12 * h
    # Compute the rest in two steps
    for _ in 2:n
        k₄ = k₂ * ϰ / norm(r)^3
        k₅ = k₃ * ϰ / norm(r₀)^3
        function g!(F, x)
            F[1:N] = -x[1:N] + r + k₁ * x[N+1:end] + k₂ * v + k₃ * v₀
            F[N+1:end] = -x[N+1:end] + v +
                k₁ * ϰ * x[1:N] / norm(x[1:N])^3 +
                k₄ * r + k₅ * r₀
        end
        res = nlsolve(g!, [r; v]; autodiff=:forward, method=:newton, ftol=1e-16).zero
        r₀, v₀ = r, v
        r, v = res[1:N], res[N+1:end]
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Moulton's method (with
# the trapezoidal rule as a starter), return
# the values of position and velocity on the
# last step
function am2(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the trapezoidal rule
    r, v = trapezoidal(r₀, v₀, h, UInt(1))
    return _am2(r₀, v₀, r, v, h, n)
end

# Integrate equations of motion using the
# two-step Adams–Moulton's method (with the
# 4th-order Runge-Kutta method as a starter),
# return the values of position and velocity
# on the last step
function am2_rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the 4th-order Runge-Kutta method
    r, v = rk4(r₀, v₀, h, UInt(1))
    return _am2(r₀, v₀, r, v, h, n)
end

# Integrate equations of motion using the
# three-step Adams–Moulton's method, return
# the values of position and velocity on the
# last step
function _am3(
    r₀::Vector{F},
    v₀::Vector{F},
    r₁::Vector{F},
    v₁::Vector{F},
    r::Vector{F},
    v::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Define independent coefficients
    k₁ = 9 / 24 * h
    k₂ = 19 / 24 * h
    k₃ = -5 / 24 * h
    k₄ = 1 / 24 * h
    # Compute the rest in two steps
    for _ in 3:n
        # Define dependent coefficients
        k₅ = k₂ * ϰ / norm(r)^3
        k₆ = k₃ * ϰ / norm(r₁)^3
        k₇ = k₄ * ϰ / norm(r₀)^3
        function g!(F, x)
            F[1:N] = -x[1:N] + r + k₁ * x[N+1:end] + k₂ * v + k₃ * v₁ + k₄ * v₀
            F[N+1:end] = -x[N+1:end] + v +
                   k₁ * ϰ * x[1:N] / norm(x[1:N])^3 +
                   k₅ * r + k₆ * r₁ + k₇ * r₀
        end
        res = nlsolve(g!, [r; v]; autodiff=:forward, method=:newton, ftol=1e-16).zero
        r₀, r₁, v₀, v₁ = r₁, r, v₁, v
        r, v = res[1:N], res[N+1:end]
    end
    return r, v
end

# Integrate equations of motion using the
# three-step Adams–Moulton's method (with
# the trapezoidal rule and the two-step
# Adams–Moulton's method as starters), return
# the values of position and velocity on the
# last step
function am3(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second value of the solution
    # using the trapezoidal rule
    r₁, v₁ = trapezoidal(r₀, v₀, h, UInt(1))
    # Compute the third value of the solution
    # using the two-step Adams–Moulton's method
    r, v = am2(r₀, v₀, h, UInt(2))
    return _am3(r₀, v₀, r₁, v₁, r, v, h, n)
end

# Integrate equations of motion using the
# three-step Adams–Moulton's method (with the
# 4th-order Runge-Kutta method as a starter),
# return the values of position and velocity
# on the last step
function am3_rk4(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Compute the second and third values of the
    # solution using the 4th-order Runge-Kutta method
    r₁, v₁ = rk4(r₀, v₀, h, UInt(1))
    r, v = rk4(r₀, v₀, h, UInt(2))
    return _am3(r₀, v₀, r₁, v₁, r, v, h, n)
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
    [
        "euler.dat",
        "ab2.dat",
        "ab2_rk4.dat",
        "ab3.dat",
        "ab3_rk4.dat",
        "ab4.dat",
        "ab4_rk4.dat",
        "rk4.dat",
        "b_euler.dat",
        "trapezoidal.dat",
        "am2.dat",
        "am2_rk4.dat",
        "am3.dat",
        "am3_rk4.dat",
    ]
)

# Prepare a list of methods
const methods = (
    euler,
    ab2,
    ab2_rk4,
    ab3,
    ab3_rk4,
    ab4,
    ab4_rk4,
    rk4,
    b_euler,
    trapezoidal,
    am2,
    am2_rk4,
    am3,
    am3_rk4,
)

# Integrate equations of motion using all available
# methods and print the values of position and velocity
# on the last step
function integrate(h::Vector{F}, n::Vector{I}) where {F <: AbstractFloat,I <: Unsigned}
    # Open the data files
    io = open.(files, "a")

    # Prepare results vectors
    r = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    v = [ Vector{Vector{F}}() for _ in 1:length(io) ]
    E = [ Vector{F}() for _ in 1:length(io) ]
    ΔE = [ Vector{F}() for _ in 1:length(io) ]

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
            Printf.format(io[j], get_format(i),  v[j][i][1], v[j][i][2])
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
    [1e-2, 1e-3, 1e-4, 1e-5], #, 1e-6, 1e-7],
    UInt.([ 10^i for i in 2:5 ]),
)

# Integrate the full cycle
integrate(
    [1e-2, 1e-3, 1e-4, 1e-5], #, 1e-6, 1e-7],
    UInt.([ 271, 2714, 27141, 271408]), #, 2714081, 27140809 ]),
)

println()
