# This script provides implementations of the
# Adams–Bashforth's methods of different orders
# for solving a system of differential equations
# and compares them to the exact solutions and
# other implementations

println('\n', " "^4, "> Loading the packages...")

# using DifferentialEquations # Other implementations
using LinearAlgebra # Norm
using Plots # Plotting

# Use the GR backend for plots
gr()

# Change the default font for plots
default(fontfamily="Computer Modern", dpi=300, legend=:topright)

# Define the value of ϰ
const ϰ = -1

# Integrate equations of motion using the
# Euler's method, return the values of position
# and velocity on the last step
function motion_euler_last(
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
    # Compute the solutions (in-place)
    for _ in 1:n
        ρ = norm(r)^3
        for k in 1:N
            a = ϰ * r[k] / ρ
            r[k] += h * v[k]
            v[k] += h * a
        end
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Bashforth's method, return
# the values of position and velocity on the
# last step
function motion_ab2_last(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Vector{F},Vector{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare vectors for the solutions
    r = copy(r₀)
    v = copy(v₀)
    # Prepare buffers for previous values
    rₖ₋₁ = copy(r₀)
    vₖ₋₁ = copy(v₀)
    # Compute the second value of the solution
    # by using the one-step Euler's method
    ρ = norm(r)^3
    for k in 1:N
        a = ϰ * r[k] / ρ
        r[k] += h * v[k]
        v[k] += h * a
    end
    # Define a couple of independent coefficients
    k₁ = 3 / 2 * h
    k₂ = 1 / 2 * h
    # Compute the rest in two steps
    for _ in 2:n
        ρ = norm(r)^3
        # Define a couple of dependent coefficients
        k₃ = k₁ * ϰ / ρ
        k₄ = k₂ * ϰ / ρ
        for k in 1:N
            a₁ = k₃ * r[k]
            a₂ = k₄ * rₖ₋₁[k]
            rₖ₋₁[k] = r[k]
            r[k] += k₁ * v[k] - k₂ * vₖ₋₁[k]
            vₖ₋₁[k] = v[k]
            v[k] += a₁ - a₂
        end
    end
    return r, v
end

println('\n', " "^4, "> Integrating the equations of motion for the last step only...")

# Define the initial values of the position and velocity
r₀ = [1.0, 0.0]
v₀ = [0.0, 0.5]

# Calculate integrals and some of the orbit parameters
E = 1 / 2 * norm(v₀)^2 - ϰ^2 / norm(r₀)
a = -ϰ^2 / 2 / E
c = r₀[1] * v₀[2] - r₀[2] * v₀[1]
p = c^2 / ϰ^2
b = sqrt(a * p)

# Calculate the ellipse
t = range(0, 2 * π; length=100)
x = a * cos.(t) .+ (r₀[1] - a)
y = b * sin.(t)

println(
    '\n',
    " "^6, "r₀: ", r₀[1], " ", r₀[2], '\n',
    " "^6, "v₀: ", v₀[1], " ", v₀[2], '\n',
    " "^6, "E: ", E,
)

# Integrate equations of motion using all available
# methods and print the values of position and velocity
# on the last step
function motion_integrate_last(h::F, n::I) where {F <: AbstractFloat,I <: Unsigned}
    rₑ, vₑ = motion_euler_last(r₀, v₀, h, n)
    rₐ₂, vₐ₂ = motion_ab2_last(r₀, v₀, h, n)
    Eₑ = 1 / 2 * norm(vₑ)^2 - ϰ^2 / norm(rₑ)
    Eₐ₂ = 1 / 2 * norm(vₐ₂)^2 - ϰ^2 / norm(rₐ₂)
    ΔEₑ = abs(E - Eₑ)
    ΔEₐ₂ = abs(E - Eₐ₂)

    println(
        '\n',
        " "^6, "h = $(h), n = $(n)", '\n',
        '\n',
        " "^6, "Euler:", '\n',
        " "^6, "r: ", rₑ[1], " ", rₑ[2], '\n',
        " "^6, "v: ", vₑ[1], " ", vₑ[2], '\n',
        " "^6, "E: ", Eₑ, '\n',
        " "^6, "ΔE: ", ΔEₑ, '\n',
        '\n',
        " "^6, "Adams–Bashforth (2-step):", '\n',
        " "^6, "r: ", rₐ₂[1], " ", rₐ₂[2], '\n',
        " "^6, "v: ", vₐ₂[1], " ", vₐ₂[2], '\n',
        " "^6, "E: ", Eₐ₂, '\n',
        " "^6, "ΔE: ", ΔEₐ₂,
    )
end

motion_integrate_last(0.01, UInt(100))
motion_integrate_last(0.001, UInt(1000))
motion_integrate_last(0.0001, UInt(10000))
motion_integrate_last(0.00001, UInt(100000))
motion_integrate_last(0.000001, UInt(1000000))
motion_integrate_last(0.0000001, UInt(10000000))
motion_integrate_last(0.01, UInt(271))
motion_integrate_last(0.001, UInt(2714))
motion_integrate_last(0.0001, UInt(27141))
motion_integrate_last(0.00001, UInt(271408))
motion_integrate_last(0.000001, UInt(2714081))
motion_integrate_last(0.0000001, UInt(27140809))

println('\n', " "^4, "> Integrating the equations of motion fully...")

# Integrate equations of motion using the
# Euler's method, return the values of position
# and velocity on each step
function motion_euler(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Matrix{F},Matrix{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare output matrixes
    r = Matrix{F}(undef, n + 1, N)
    v = Matrix{F}(undef, n + 1, N)
    r[1, :] = copy(r₀)
    v[1, :] = copy(v₀)
    # Compute the solutions
    for i in 2:(n + 1)
        ρ = norm(r[i - 1, :])^3
        for k in 1:N
            r[i, k] = r[i - 1, k] + h * v[i - 1, k]
            v[i, k] = v[i - 1, k] + h * ϰ * r[i - 1, k] / ρ
        end
    end
    return r, v
end

# Integrate equations of motion using the
# two-step Adams–Bashforth's method, return
# the values of position and velocity on each step
function motion_ab2(
    r₀::Vector{F},
    v₀::Vector{F},
    h::F,
    n::I,
)::Tuple{Matrix{F},Matrix{F}} where
{F <: AbstractFloat,I <: Unsigned}
    # Determine the length of the input vectors
    N = length(r₀)
    # Prepare output matrixes
    r = Matrix{F}(undef, n + 1, N)
    v = Matrix{F}(undef, n + 1, N)
    r[1, :] = copy(r₀)
    v[1, :] = copy(v₀)
    # Compute the second value of the solution
    # by using the one-step Euler's method
    ρ = norm(r[1, :])^3
    for k in 1:N
        r[2, k] = r[1, k] + h * v[1, k]
        v[2, k] = v[1, k] + h * ϰ * r[1, k] / ρ
    end
    # Define a couple of independent coefficients
    k₁ = 3 / 2 * h
    k₂ = 1 / 2 * h
    # Compute the rest in two steps
    for i in 3:(n + 1)
        ρ = norm(r[i - 1, :])^3
        # Define a couple of dependent coefficients
        k₃ = k₁ * ϰ / ρ
        k₄ = k₂ * ϰ / ρ
        for k in 1:N
            r[i, k] = r[i - 1, k] + k₁ * v[i - 1, k] - k₂ * v[i - 2, k]
            v[i, k] = v[i - 1, k] + k₃ * r[i - 1, k] - k₄ * r[i - 2, k]
        end
    end
    return r, v
end

# Integrate equations of motion using all available
# methods and create a comparison plot between the solutions
function motion_integrate(h::F, n::I) where {F <: AbstractFloat,I <: Unsigned}
    rₑ, _ = motion_euler(r₀, v₀, h, n)
    rₐ₂, _ = motion_ab2(r₀, v₀, h, n)

    # Plot the exact solution
    p = plot(
        x,
        y;
        label="Т",
        xlabel="x",
        ylabel="y",
        xlims=extrema(x) .+ (-0.1, 0.1),
        ylims=extrema(y) .+ (-0.1, 0.1),
    );

    # Add the solution achieved by
    # the Euler's method to the plot
    plot!(p, rₑ[:, 1], rₑ[:, 2]; label="Э");

    # Add the solution achieved by the two-step
    # Adams–Bashforth's method to the plot
    plot!(p, rₐ₂[:, 1], rₐ₂[:, 2]; label="АБ-2");

    # Save the figure
    savefig(p, "$(@__DIR__)/../plots/orbit_$(h).pdf")

    println(" "^6, "* The figure `orbit_$(h).pdf` is saved. *")

    # Create an animation of the same thing but with moving circles
    anim = @animate for i in 1:10^(length("$(n)") - 3):n
        # Recreate the static plot
        plot(
            x,
            y;
            label="Т",
            xlabel="x",
            ylabel="y",
            xlims=extrema(x) .+ (-0.1, 0.1),
            ylims=extrema(y) .+ (-0.1, 0.1),
        );
        plot!(rₑ[:, 1], rₑ[:, 2]; label="Э")
        plot!(rₐ₂[:, 1], rₐ₂[:, 2]; label="АБ-2")
        # Calculate the point on the ellipse
        tₚ = 2 * π * (i - 1) / n
        xₚ = a * cos(tₚ) + (r₀[1] - a)
        yₚ = b * sin(tₚ)
        # Add moving points
        scatter!([xₚ,], [yₚ,]; label="")
        scatter!([rₑ[i, 1],], [rₑ[i, 2],]; label="")
        scatter!([rₐ₂[i, 1],], [rₐ₂[i, 2],]; label="")
    end

    # Save the animation
    gif(anim, "$(@__DIR__)/../plots/anim_orbit_$(h).gif"; fps=30, show_msg=false)

    println(" "^6, "* The animation `anim_orbit_$(h).pdf` is saved. *")

end

println()
motion_integrate(0.01, UInt(271))
motion_integrate(0.001, UInt(2714))
motion_integrate(0.0001, UInt(27141))
motion_integrate(0.00001, UInt(271408))

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
        ρ₁₂ = norm(r₁[i - 1, :] - r₂[i - 1, :])^3
        ρ₁₃ = norm(r₁[i - 1, :] - r₃[i - 1, :])^3
        ρ₂₃ = norm(r₂[i - 1, :] - r₃[i - 1, :])^3
        # Define a couple of dependent coefficients
        for k in 1:N
            r₁[i, k] = r₁[i - 1, k] + k₁ * v₁[i - 1, k] - k₂ * v₁[i - 2, k]
            v₁[i, k] = v₁[i - 1, k] + k₁ * ϰ * (r₁[i - 1, k] - r₂[i - 1, k]) / ρ₁₂ +
                                      k₁ * ϰ * (r₁[i - 1, k] - r₃[i - 1, k]) / ρ₁₃ -
                                      k₂ * ϰ * (r₁[i - 2, k] - r₂[i - 2, k]) / ρ₁₂ -
                                      k₂ * ϰ * (r₁[i - 2, k] - r₃[i - 2, k]) / ρ₁₃
            r₂[i, k] = r₂[i - 1, k] + k₁ * v₂[i - 1, k] - k₂ * v₂[i - 2, k]
            v₂[i, k] = v₂[i - 1, k] + k₁ * ϰ * (r₂[i - 1, k] - r₁[i - 1, k]) / ρ₁₂ +
                                      k₁ * ϰ * (r₂[i - 1, k] - r₃[i - 1, k]) / ρ₂₃ -
                                      k₂ * ϰ * (r₂[i - 2, k] - r₁[i - 2, k]) / ρ₁₂ -
                                      k₂ * ϰ * (r₂[i - 2, k] - r₃[i - 2, k]) / ρ₂₃
            r₃[i, k] = r₃[i - 1, k] + k₁ * v₃[i - 1, k] - k₂ * v₃[i - 2, k]
            v₃[i, k] = v₃[i - 1, k] + k₁ * ϰ * (r₃[i - 1, k] - r₁[i - 1, k]) / ρ₁₃ +
                                      k₁ * ϰ * (r₃[i - 1, k] - r₂[i - 1, k]) / ρ₂₃ -
                                      k₂ * ϰ * (r₃[i - 2, k] - r₁[i - 2, k]) / ρ₁₃ -
                                      k₂ * ϰ * (r₃[i - 2, k] - r₂[i - 2, k]) / ρ₂₃
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
