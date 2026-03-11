/// Gaussian kernel smoothing of a 1D signal, replacing R's ksmooth.
///
/// * `x` - positions (sorted)
/// * `y` - values at positions
/// * `x_out` - output positions to evaluate at
/// * `bandwidth` - kernel bandwidth (standard deviation)
pub fn gaussian_kernel_smooth(x: &[f64], y: &[f64], x_out: &[f64], bandwidth: f64) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let bw2 = 2.0 * bandwidth * bandwidth;

    x_out
        .iter()
        .map(|&xo| {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;

            for (xi, yi) in x.iter().zip(y.iter()) {
                let d = xo - xi;
                let w = (-d * d / bw2).exp();
                weight_sum += w;
                value_sum += w * yi;
            }

            if weight_sum > 0.0 {
                value_sum / weight_sum
            } else {
                0.0
            }
        })
        .collect()
}

/// Find local modes (peaks) in a smoothed signal.
/// Returns indices where y[i] > y[i-1] and y[i] > y[i+1].
pub fn find_modes(y: &[f64]) -> Vec<usize> {
    if y.len() < 3 {
        return Vec::new();
    }

    let mut modes = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            modes.push(i);
        }
    }
    modes
}

/// Find local valleys (minima) in a smoothed signal.
/// Returns indices where y[i] < y[i-1] and y[i] < y[i+1].
pub fn find_valleys(y: &[f64]) -> Vec<usize> {
    if y.len() < 3 {
        return Vec::new();
    }

    let mut valleys = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] < y[i - 1] && y[i] < y[i + 1] {
            valleys.push(i);
        }
    }
    valleys
}

/// Compute a coverage histogram from fragment positions.
/// Returns (positions, counts) at the given resolution.
pub fn coverage_histogram(
    positions: &[f64],
    utr_length: f64,
    resolution: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_bins = (utr_length / resolution).ceil() as usize;
    let mut counts = vec![0.0; n_bins];

    for &pos in positions {
        let bin = ((pos - 1.0) / resolution) as usize;
        if bin < n_bins {
            counts[bin] += 1.0;
        }
    }

    let positions: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * resolution).collect();

    (positions, counts)
}
