#!/usr/bin/env julia
# ╭─────────────────────────────────────────────────────────────────────╮
# │ RainSynth.jl                                                        │
# │ Ported and refined from a SciPy reference version                   │
# ╰─────────────────────────────────────────────────────────────────────╯

module RainSynth

using Random, DSP, WAV

# ╶────────────────────────────────────────────────╴ [ helper utilities ]

"""
    butter_lowpass(cutoff, fs; order = 6)

Return a closure that applies an `order`-th order Butterworth low-pass
filter with cut-off frequency `cutoff` (Hz) to data sampled at `fs` (Hz).
"""
function butter_lowpass(cutoff::Real, fs::Real; order::Int = 6)
    H = digitalfilter(Lowpass(cutoff, fs = fs), Butterworth(order))
    x -> filt(H, x) # closure
end

"""
    pink_noise(n)

Generate `n` samples of approximate 1/f (pink) noise using the
Voss–McCartney algorithm.
"""
function pink_noise(n::Integer)
    rows = 16
    data = cumsum(randn(rows, n), dims = 2)
    vec(view(data, rows, :)) ./ rows
end

"""
    exponential_decay(len, fs; τ = 0.015)

Return a vector of length `len` representing an exponential decay with
time constant `τ` (s) at sampling rate `fs` (Hz).
"""
exponential_decay(len::Integer, fs::Real; τ::Float64 = 0.015) =
    @. exp(-(0:len-1)/(fs * τ))

"""
    random_pan(signal) → 2 × N matrix

Constant-power panning of monophonic `signal`.
The result has two rows (left, right) and `length(signal)` columns.
"""
function random_pan(signal::AbstractVector{T}) where {T<:Real}
    θ   = rand() * (π/2)                          # 0 ⇒ left, π/2 ⇒ right
    out = Array{T}(undef, 2, length(signal))
    @inbounds begin
        out[1, :] .=  cos(θ) .* signal
        out[2, :] .=  sin(θ) .* signal
    end
    return out
end

# ╶────────────────────────────────────────────────╴ [ main synthesizer ]

"""
    synth_rain(; duration = 30.0, fs = 44_100, intensity = 1.0)

Generate a synthetic rain signal of length `duration` seconds at sampling
rate `fs` (Hz).
`intensity` scales the expected droplet rate
(≈ 120 drops · s⁻¹ when `intensity = 1`).

Returns an array of size `(samples, channels)` (column-major, WAV-ready).
"""
function synth_rain(; duration::Real  = 30.0,
                      fs::Integer     = 44_100,
                      intensity::Real = 1.0)

    nsamp = Int(round(duration * fs))

    # ╶─────────────────────────────────────────────╴ [ background hiss ]
    white = randn(nsamp)
    hiss  = 0.15 .* butter_lowpass(8_000, fs)(white)

    n_knots   = 8
    knot_idx  = round.(Int, LinRange(1, nsamp, n_knots))
    knot_val  = rand(n_knots)
    lfo       = similar(hiss)

    for k in 1:n_knots-1
        i, j = knot_idx[k], knot_idx[k+1]
        lfo[i:j] .= range(knot_val[k], knot_val[k+1]; length = j - i + 1)
    end
    hiss .*= 0.7 .+ 0.6 .* lfo

    left   = hiss .* (0.9 + 0.2 * rand())
    right  = hiss .* (0.9 + 0.2 * rand())
    stereo = [left'; right'] # 2 × nsamp

    # ╶──────────────────────────────────────────────╴ [ droplet grains ]

    λ         = 100 * intensity                        # drops per second
    n_drops   = Int(round(λ * duration))
    positions = rand(1 : nsamp - 2_000, n_drops)

    for pos in positions
        len   = rand(fs ÷ 1_000 : fs ÷ 80)
        grain = pink_noise(len) .* exponential_decay(len, fs)
        grain .*= rand(0.2:0.001:1.0)
        g_st  = random_pan(grain)                               # 2 × len

        stop = min(pos + len - 1, nsamp)
        stereo[:, pos:stop] .+= g_st[:, 1:(stop - pos + 1)]
    end

    # ╶───────────────────────────────────────────────╴ [ normalization ]
    peak = maximum(abs, stereo)
    stereo ./= peak * 1.05

    return stereo' # (samples, channels)
end


# ╶──────────────────────────────────────────────────────╴ [ write file ]
"""
    write_rain(filename; duration = 30.0, fs = 44_100, intensity = 1.0)

Synthesize a rain signal and write it to `filename` (WAV).
"""
function write_rain(filename::AbstractString;
                    duration  = 30.0,
                    fs        = 44_100,
                    intensity = 1.0)
    audio = synth_rain(duration = duration, fs = fs, intensity = intensity)
    wavwrite(audio, filename; Fs = fs)
    return filename
end

end # module RainSynth

# ╶───────────────────────────────────────────────────╴ [ main driver ]
if abspath(PROGRAM_FILE) == @__FILE__
    fname = RainSynth.write_rain("synthetic_rain.wav"; duration = 30.0)
    println("File written: ", fname)
end
