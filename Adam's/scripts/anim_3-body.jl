# This script provides implementations of the
# Adams–Bashforth's methods of different orders
# for solving a system of differential equations
# and compares them to the exact solutions and
# other methods

println('\n', " "^4, "> Loading the packages...")

using LinearAlgebra # Norm
using Plots # Plotting

# Use the GR backend for plots
gr()

# Change the default font for plots
default(fontfamily="Computer Modern", dpi=300, legend=:topright)

# Define the value of ϰ
const ϰ = -1

# Integrate the three-body problem using the
# Euler's method, return the values of position
# and velocity on each step
function motion3_euler(
    r₀₁::Vector{F},
    r₀₂::Vector{F},
    r₀₃::Vector{F},
    v₀₁::Vector{F},
    v₀₂::Vector{F},
    v₀₃::Vector{F},
    h::F,
    n::I,
)::Tuple{Matrix{F},Matrix{F},Matrix{F},Matrix{F},Matrix{F},Matrix{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare output matrixes
    r₁ = Matrix{F}(undef, n + 1, N)
    r₂ = Matrix{F}(undef, n + 1, N)
    r₃ = Matrix{F}(undef, n + 1, N)
    v₁ = Matrix{F}(undef, n + 1, N)
    v₂ = Matrix{F}(undef, n + 1, N)
    v₃ = Matrix{F}(undef, n + 1, N)
    r₁[1, :] = copy(r₀₁)
    r₂[1, :] = copy(r₀₂)
    r₃[1, :] = copy(r₀₃)
    v₁[1, :] = copy(v₀₁)
    v₂[1, :] = copy(v₀₂)
    v₃[1, :] = copy(v₀₃)
    # Compute the solutions
    for i in 2:(n + 1)
        ρ₁₂ = norm(r₁[i - 1, :] - r₂[i - 1, :])^3
        ρ₁₃ = norm(r₁[i - 1, :] - r₃[i - 1, :])^3
        ρ₂₃ = norm(r₂[i - 1, :] - r₃[i - 1, :])^3
        for k in 1:N
            r₁[i, k] = r₁[i - 1, k] + h * v₁[i - 1, k]
            v₁[i, k] = v₁[i - 1, k] + h * ϰ * (r₁[i - 1, k] - r₂[i - 1, k]) / ρ₁₂ +
                                      h * ϰ * (r₁[i - 1, k] - r₃[i - 1, k]) / ρ₁₃
            r₂[i, k] = r₂[i - 1, k] + h * v₂[i - 1, k]
            v₂[i, k] = v₂[i - 1, k] + h * ϰ * (r₂[i - 1, k] - r₁[i - 1, k]) / ρ₁₂ +
                                      h * ϰ * (r₂[i - 1, k] - r₃[i - 1, k]) / ρ₂₃
            r₃[i, k] = r₃[i - 1, k] + h * v₃[i - 1, k]
            v₃[i, k] = v₃[i - 1, k] + h * ϰ * (r₃[i - 1, k] - r₁[i - 1, k]) / ρ₁₃ +
                                      h * ϰ * (r₃[i - 1, k] - r₂[i - 1, k]) / ρ₂₃
        end
    end
    return r₁, r₂, r₃, v₁, v₂, v₃
end

# Integrate the three-body problem using the
# two-step Adams–Bashforth's method, return the
# values of position and velocity on each step
function motion3_ab2(
    r₀₁::Vector{F},
    r₀₂::Vector{F},
    r₀₃::Vector{F},
    v₀₁::Vector{F},
    v₀₂::Vector{F},
    v₀₃::Vector{F},
    h::F,
    n::I,
)::Tuple{Matrix{F},Matrix{F},Matrix{F},Matrix{F},Matrix{F},Matrix{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare output matrixes
    r₁ = Matrix{F}(undef, n + 1, N)
    r₂ = Matrix{F}(undef, n + 1, N)
    r₃ = Matrix{F}(undef, n + 1, N)
    v₁ = Matrix{F}(undef, n + 1, N)
    v₂ = Matrix{F}(undef, n + 1, N)
    v₃ = Matrix{F}(undef, n + 1, N)
    r₁[1, :] = copy(r₀₁)
    r₂[1, :] = copy(r₀₂)
    r₃[1, :] = copy(r₀₃)
    v₁[1, :] = copy(v₀₁)
    v₂[1, :] = copy(v₀₂)
    v₃[1, :] = copy(v₀₃)
    # Compute the second value of the solution
    # by using the one-step Euler's method
    ρ₁₂ = norm(r₁[1, :] - r₂[1, :])^3
    ρ₁₃ = norm(r₁[1, :] - r₃[1, :])^3
    ρ₂₃ = norm(r₂[1, :] - r₃[1, :])^3
    for k in 1:N
        r₁[2, k] = r₁[1, k] + h * v₁[1, k]
        v₁[2, k] = v₁[1, k] + h * ϰ * (r₁[1, k] - r₂[1,k]) / ρ₁₂ +
                              h * ϰ * (r₁[1, k] - r₃[1,k]) / ρ₁₃
        r₂[2, k] = r₂[1, k] + h * v₂[1, k]
        v₂[2, k] = v₂[1, k] + h * ϰ * (r₂[1, k] - r₁[1,k]) / ρ₁₂ +
                              h * ϰ * (r₂[1, k] - r₃[1,k]) / ρ₂₃
        r₃[2, k] = r₃[1, k] + h * v₃[1, k]
        v₃[2, k] = v₃[1, k] + h * ϰ * (r₃[1, k] - r₁[1,k]) / ρ₁₃ +
                              h * ϰ * (r₃[1, k] - r₂[1,k]) / ρ₂₃
    end
    # Define a couple of independent coefficients
    k₁ = 3 / 2 * h
    k₂ = 1 / 2 * h
    # Compute the rest in two steps
    for i in 3:(n + 1)
        ρ₁₂_ₖ = norm(r₁[i - 1, :] - r₂[i - 1, :])^3
        ρ₁₃_ₖ = norm(r₁[i - 1, :] - r₃[i - 1, :])^3
        ρ₂₃_ₖ = norm(r₂[i - 1, :] - r₃[i - 1, :])^3
        ρ₁₂_ₖ₋₁ = norm(r₁[i - 2, :] - r₂[i - 2, :])^3
        ρ₁₃_ₖ₋₁ = norm(r₁[i - 2, :] - r₃[i - 2, :])^3
        ρ₂₃_ₖ₋₁ = norm(r₂[i - 2, :] - r₃[i - 2, :])^3
        # Define a couple of dependent coefficients
        for k in 1:N
            r₁[i, k] = r₁[i - 1, k] + k₁ * v₁[i - 1, k] - k₂ * v₁[i - 2, k]
            v₁[i, k] = v₁[i - 1, k] + k₁ * ϰ * (r₁[i - 1, k] - r₂[i - 1, k]) / ρ₁₂_ₖ +
                                      k₁ * ϰ * (r₁[i - 1, k] - r₃[i - 1, k]) / ρ₁₃_ₖ -
                                      k₂ * ϰ * (r₁[i - 2, k] - r₂[i - 2, k]) / ρ₁₂_ₖ₋₁ -
                                      k₂ * ϰ * (r₁[i - 2, k] - r₃[i - 2, k]) / ρ₁₃_ₖ₋₁
            r₂[i, k] = r₂[i - 1, k] + k₁ * v₂[i - 1, k] - k₂ * v₂[i - 2, k]
            v₂[i, k] = v₂[i - 1, k] + k₁ * ϰ * (r₂[i - 1, k] - r₁[i - 1, k]) / ρ₁₂_ₖ +
                                      k₁ * ϰ * (r₂[i - 1, k] - r₃[i - 1, k]) / ρ₂₃_ₖ -
                                      k₂ * ϰ * (r₂[i - 2, k] - r₁[i - 2, k]) / ρ₁₂_ₖ₋₁ -
                                      k₂ * ϰ * (r₂[i - 2, k] - r₃[i - 2, k]) / ρ₂₃_ₖ₋₁
            r₃[i, k] = r₃[i - 1, k] + k₁ * v₃[i - 1, k] - k₂ * v₃[i - 2, k]
            v₃[i, k] = v₃[i - 1, k] + k₁ * ϰ * (r₃[i - 1, k] - r₁[i - 1, k]) / ρ₁₃_ₖ +
                                      k₁ * ϰ * (r₃[i - 1, k] - r₂[i - 1, k]) / ρ₂₃_ₖ -
                                      k₂ * ϰ * (r₃[i - 2, k] - r₁[i - 2, k]) / ρ₁₃_ₖ₋₁ -
                                      k₂ * ϰ * (r₃[i - 2, k] - r₂[i - 2, k]) / ρ₂₃_ₖ₋₁
        end
    end
    return r₁, r₂, r₃, v₁, v₂, v₃
end

println('\n', " "^4, "> Integrating the three-body problem...")

# Define the initial values of the position and velocity
r₀₁ = [0.97, -0.2431]
r₀₂ = [-0.97, 0.2431]
r₀₃ = [0.0, 0.0]
v₀₁ = [0.4662, 0.4324]
v₀₂ = [0.4662, 0.4324]
v₀₃ = [-0.9324, -0.8647]

@userplot CirclePlot
@recipe function f(cp::CirclePlot)
    x, y, i = cp.args
    n = length(x)
    inds = circshift(1:n, 1 - i)
    linewidth --> range(0, 10, length=n)
    seriesalpha --> range(0, 1, length=n)
    aspect_ratio --> 1
    label --> false
    x[inds], y[inds]
end

h = 0.0001
n = UInt(62887)

r₁ₑ, r₂ₑ, r₃ₑ, _, _, _ = motion3_ab2(r₀₁, r₀₂, r₀₃, v₀₁, v₀₂, v₀₃, h, n)

# Animate the figure-8 orbit
anim = @animate for i ∈ 1:446:n
    circleplot(r₁ₑ[:, 1], r₁ₑ[:, 2], i; xlabel="x", ylabel="y")
    circleplot!(r₂ₑ[:, 1], r₂ₑ[:, 2], i)
    circleplot!(r₃ₑ[:, 1], r₃ₑ[:, 2], i)
end

# Save the animation
gif(anim, "$(@__DIR__)/../plots/anim_3body.gif"; fps=30, show_msg=false)

println(" "^6, "* The animation `anim_3body.gif` is saved. *")

println()
