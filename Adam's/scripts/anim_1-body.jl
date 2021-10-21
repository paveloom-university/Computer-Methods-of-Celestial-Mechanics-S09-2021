println('\n', " "^4, "> Loading the packages...")

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
    k₂ = -1 / 2 * h
    # Compute the rest in two steps
    for i in 3:(n + 1)
        ρₖ = norm(r[i - 1, :])^3
        ρₖ₋₁ = norm(r[i - 2, :])^3
        # Define a couple of dependent coefficients
        k₃ = k₁ * ϰ / ρₖ
        k₄ = k₂ * ϰ / ρₖ₋₁
        for k in 1:N
            r[i, k] = r[i - 1, k] + k₁ * v[i - 1, k] + k₂ * v[i - 2, k]
            v[i, k] = v[i - 1, k] + k₃ * r[i - 1, k] + k₄ * r[i - 2, k]
        end
    end
    return r, v
end

# Integrate equations of motion using all available
# methods and create a comparison plot between the solutions
function motion_integrate(rₐ::Matrix{F}, r₀::Vector{F}, v₀::Vector{F}, h::F, n::I) where {F <: AbstractFloat,I <: Unsigned}
    rₑ, _ = motion_euler(r₀, v₀, h, n)
    rₐ₂, _ = motion_ab2(r₀, v₀, h, n)

    # Plot the exact solution
    p = plot(
        rₐ[1:1000:end, 1],
        rₐ[1:1000:end, 2],
        label="Т",
        xlabel="x",
        ylabel="y",
        xlims=extrema(rₐ[:, 1]) .+ (-0.1, 0.1),
        ylims=extrema(rₐ[:, 2]) .+ (-0.1, 0.1),
    );

    # Add the solution achieved by
    # the Euler's method to the plot
    plot!(p, rₑ[:, 1], rₑ[:, 2]; label="Э");

    # Add the solution achieved by the two-step
    # Adams–Bashforth's method to the plot
    plot!(p, rₐ₂[:, 1], rₐ₂[:, 2]; label="АБ-2");

    # Save the figure
    savefig(p, "$(@__DIR__)/../plots/static/orbit_$(h).png")
    savefig(p, "$(@__DIR__)/../plots/static/orbit_$(h).pdf")

    println(" "^6, "* The figures `orbit_$(h).png` and `orbit_$(h).pdf` are saved. *")

    # Create an animation of the same thing but with moving circles
    anim = @animate for i in 1:10^(length("$(n)") - 3):n
        # Recreate the static plot
        plot(
            rₐ[1:1000:end, 1],
            rₐ[1:1000:end, 2],
            label="Т",
            xlabel="x",
            ylabel="y",
            xlims=extrema(rₐ[:, 1]) .+ (-0.1, 0.1),
            ylims=extrema(rₐ[:, 2]) .+ (-0.1, 0.1),
        );
        plot!(rₑ[:, 1], rₑ[:, 2]; label="Э")
        plot!(rₐ₂[:, 1], rₐ₂[:, 2]; label="АБ-2")
        # Add moving points
        scatter!([rₑ[i, 1],], [rₑ[i, 2],]; label="")
        scatter!([rₐ₂[i, 1],], [rₐ₂[i, 2],]; label="")
    end

    # Save the animation
    gif(anim, "$(@__DIR__)/../plots/animated/orbit_$(h).gif"; fps=30, show_msg=false)

    println(" "^6, "* The animation `orbit_$(h).gif` is saved. *")

end

println(" "^4, "> Integrating the equations of motion...")

# Define the initial values of the position and velocity
r₀ = [1.0, 0.0]
v₀ = [0.0, 0.5]

# Calculate the (close-to) analytical solution
rₐ, _ = motion_ab2(r₀, v₀, 0.000001, UInt(2714081))

println()
motion_integrate(rₐ, r₀, v₀, 0.01, UInt(271))
motion_integrate(rₐ, r₀, v₀, 0.001, UInt(2714))
motion_integrate(rₐ, r₀, v₀, 0.0001, UInt(27141))
motion_integrate(rₐ, r₀, v₀, 0.00001, UInt(271408))
println()
