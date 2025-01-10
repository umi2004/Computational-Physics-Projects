import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.io.wavfile import read, write
from numpy import empty

class AudioFilter:
    def __init__(self, filename, cutoff_freq=880):
        self.filename = filename
        self.cutoff_freq = cutoff_freq
        self.sample_rate, self.data = read(filename)
        self.channel_0 = self.data[:, 0]
        self.channel_1 = self.data[:, 1]
        self.nsamples = len(self.channel_0)
        self.dt = 1 / self.sample_rate
        self.t = np.arange(self.nsamples) * self.dt
        self.frequencies = np.fft.fftfreq(self.nsamples, self.dt)

    def plot_time_series(self):
        # Plot for Channel 0
        plt.plot(self.t, self.channel_0)
        plt.title("Time Series of Channel 0")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig("TimeSeries_Ch0.png")
        print("Saved: TimeSeries_Ch0.png")
        plt.clf()

        # Plot for Channel 1
        plt.plot(self.t, self.channel_1)
        plt.title("Time Series of Channel 1")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig("TimeSeries_Ch1.png")
        print("Saved: TimeSeries_Ch1.png")
        plt.clf()

    def apply_low_pass_filter(self):
        # Apply FFT to both channels
        channel_0_fft = fft(self.channel_0)
        channel_1_fft = fft(self.channel_1)

        # Apply low-pass filter by zeroing out frequencies > cutoff_freq
        filtered_channel_0_fft = np.copy(channel_0_fft)
        filtered_channel_0_fft[np.abs(self.frequencies) > self.cutoff_freq] = 0

        filtered_channel_1_fft = np.copy(channel_1_fft)
        filtered_channel_1_fft[np.abs(self.frequencies) > self.cutoff_freq] = 0

        # Perform inverse FFT to get filtered signals
        self.channel_0_filtered = ifft(filtered_channel_0_fft).real
        self.channel_1_filtered = ifft(filtered_channel_1_fft).real

    def plot_fourier_and_time_series(self):
        # Plotting Fourier Coefficients and Time Series for Channel 0
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))

        # Original Fourier Coefficients (Amplitude Spectrum)
        axs[0].plot(self.frequencies[:self.nsamples // 2], np.abs(fft(self.channel_0))[:self.nsamples // 2])
        axs[0].set_xlim(0, 1000)
        axs[0].set_title("Channel 0: Original Fourier Coefficients")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Magnitude")

        # Filtered Fourier Coefficients (Amplitude Spectrum)
        axs[1].plot(self.frequencies[:self.nsamples // 2], np.abs(fft(self.channel_0_filtered))[:self.nsamples // 2])
        axs[1].set_xlim(0, 1000)
        axs[1].set_title("Channel 0: Filtered Fourier Coefficients")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Magnitude")

        # Original Time Series
        axs[2].plot(self.t, self.channel_0)
        axs[2].set_title("Channel 0: Original Time Series")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Amplitude")

        # Filtered Time Series
        axs[3].plot(self.t, self.channel_0_filtered)
        axs[3].set_title("Channel 0: Filtered Time Series")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Amplitude")

        plt.tight_layout()
        plt.savefig("Og_Filtered_Fourier_Coeff_Ch0.png")
        print("Saved: Og_Filtered_Fourier_Coeff_Ch0.png")
        plt.clf()

        # Plotting Fourier Coefficients and Time Series for Channel 1
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))

        # Original Fourier Coefficients (Amplitude Spectrum)
        axs[0].plot(self.frequencies[:self.nsamples // 2], np.abs(fft(self.channel_1))[:self.nsamples // 2])
        axs[0].set_xlim(0, 1000)
        axs[0].set_title("Channel 1: Original Fourier Coefficients")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Magnitude")

        # Filtered Fourier Coefficients (Amplitude Spectrum)
        axs[1].plot(self.frequencies[:self.nsamples // 2], np.abs(fft(self.channel_1_filtered))[:self.nsamples // 2])
        axs[1].set_xlim(0, 1000)
        axs[1].set_title("Channel 1: Filtered Fourier Coefficients")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Magnitude")

        # Original Time Series
        axs[2].plot(self.t, self.channel_1)
        axs[2].set_title("Channel 1: Original Time Series")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Amplitude")

        # Filtered Time Series
        axs[3].plot(self.t, self.channel_1_filtered)
        axs[3].set_title("Channel 1: Filtered Time Series")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Amplitude")

        plt.tight_layout()
        plt.savefig("Og_Filtered_Fourier_Coeff_Ch1.png")
        print("Saved: Og_Filtered_Fourier_Coeff_Ch1.png")
        plt.clf()

    def compare_segments(self, duration=0.030):
        # Compare Original and Filtered Signals for a given segment duration (default: 30 ms)
        num_points = int(duration * self.sample_rate)
        t_segment = self.t[:num_points]

        # Channel 0 Segment Comparison
        plt.figure(figsize=(10, 3))
        plt.plot(t_segment, self.channel_0[:num_points], label='Original')
        plt.plot(t_segment, self.channel_0_filtered[:num_points], label='Filtered', color='orange')
        plt.title("Channel 0: Filtered Time Series ({} ms Segment, Frequencies > {} Hz Removed)".format(int(duration * 1000), self.cutoff_freq))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("Filtered_Time_Series_30ms_Ch0.png")
        print("Saved: Filtered_Time_Series_30ms_Ch0.png")
        plt.clf()

        # Channel 1 Segment Comparison
        plt.figure(figsize=(10, 3))
        plt.plot(t_segment, self.channel_1[:num_points], label='Original')
        plt.plot(t_segment, self.channel_1_filtered[:num_points], label='Filtered', color='orange')
        plt.title("Channel 1: Filtered Time Series ({} ms Segment, Frequencies > {} Hz Removed)".format(int(duration * 1000), self.cutoff_freq))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("Filtered_Time_Series_30ms_Ch1.png")
        print("Saved: Filtered_Time_Series_30ms_Ch1.png")
        plt.clf()

    def save_filtered_audio(self, output_filename='GraviteaTime_filtered.wav'):
        # Write the filtered audio to a new file
        data_out = empty(self.data.shape, dtype=self.data.dtype)
        data_out[:, 0] = self.channel_0_filtered
        data_out[:, 1] = self.channel_1_filtered

        write(output_filename, self.sample_rate, data_out)
        print(f"Saved filtered audio: {output_filename}")


def main():
    audio_filter = AudioFilter('GraviteaTime.wav')
    audio_filter.plot_time_series()
    audio_filter.apply_low_pass_filter()
    audio_filter.plot_fourier_and_time_series()
    audio_filter.compare_segments()
    audio_filter.save_filtered_audio()


if __name__ == "__main__":
    main()
