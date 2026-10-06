//! Unified progress reporting for CLI (indicatif), NAPI (JavaScript callbacks), or no-op

#[cfg(any(feature = "progress", feature = "napi"))]
pub struct ProgressReporter {
    #[cfg(feature = "progress")]
    pb: Option<indicatif::ProgressBar>,
    #[cfg(feature = "napi")]
    total: usize,
    #[cfg(feature = "napi")]
    current: usize,
    #[cfg(feature = "napi")]
    phase: String,
}

#[cfg(any(feature = "progress", feature = "napi"))]
impl ProgressReporter {
    pub fn new(phase: &str, total: usize) -> Self {
        #[cfg(feature = "progress")]
        let pb = if total > 0 {
            let pb = indicatif::ProgressBar::new(total as u64);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{msg} [{bar:40}] {pos}/{len}")
                    .unwrap(),
            );
            pb.set_message(phase.to_string());
            Some(pb)
        } else {
            None
        };

        Self {
            #[cfg(feature = "progress")]
            pb,
            #[cfg(feature = "napi")]
            total,
            #[cfg(feature = "napi")]
            current: 0,
            #[cfg(feature = "napi")]
            phase: phase.to_string(),
        }
    }

    pub fn inc(&mut self, n: usize) {
        #[cfg(feature = "progress")]
        if let Some(ref pb) = self.pb {
            pb.inc(n as u64);
        }

        #[cfg(feature = "napi")]
        {
            self.current += n;
            if self.total > 0 {
                let progress = (self.current as f64) / (self.total as f64);
                crate::napi::progress_update(progress, &self.phase);
            }
        }
    }

    pub fn finish(&self) {
        #[cfg(feature = "progress")]
        if let Some(ref pb) = self.pb {
            pb.finish_and_clear();
        }

        #[cfg(feature = "napi")]
        crate::napi::progress_update(1.0, &self.phase);
    }
}

// No-op implementation when neither feature is enabled
#[cfg(not(any(feature = "progress", feature = "napi")))]
pub struct ProgressReporter;

#[cfg(not(any(feature = "progress", feature = "napi")))]
impl ProgressReporter {
    #[inline]
    pub fn new(_phase: &str, _total: usize) -> Self {
        Self
    }

    #[inline]
    pub fn inc(&mut self, _n: usize) {}

    #[inline]
    pub fn finish(&self) {}
}
