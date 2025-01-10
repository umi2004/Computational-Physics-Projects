import numpy as np
import matplotlib.pyplot as plt

class SP500Analysis:
    def __init__(self, filename, cutoff_frequency=63):
        self.filename = filename
        self.cutoff_frequency = cutoff_frequency
        self.SP500_open_val = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=1)
        self.business_days = np.arange(len(self.SP500_open_val))
        self.SP500_fft = np.fft.rfft(self.SP500_open_val)

    def plot_opening_values(self):
        # Plot the opening value against business day number
        plt.plot(self.business_days, self.SP500_open_val)
        plt.xlabel('Business Day Number')
        plt.ylabel('S&P 500 Opening Value')
        plt.title('S&P 500 Opening Value vs Business Day Number')
        plt.grid(True)
        plt.savefig("SP500_OpenVal_BusinessDay.png")
        print("Saved: SP500_OpenVal_BusinessDay.png")
        plt.clf()

    def plot_reconstructed_data(self):
        # Reconstruct data using inverse FFT
        SP500_ifft = np.fft.irfft(self.SP500_fft)

        plt.plot(self.SP500_open_val, label="Original Data", color='blue')
        plt.plot(SP500_ifft, label="Reconstructed Data", linestyle='--', color='red')
        plt.xlabel('Business Day Number')
        plt.ylabel('S&P 500 Opening Value')
        plt.title('Comparison of Original and Reconstructed Data')
        plt.legend()
        plt.grid(True)
        plt.savefig("SP500_Og_Recon.png")
        print("Saved: SP500_Og_Recon.png")
        plt.clf()

    def plot_filtered_reconstructed_data(self):
        # Apply low-pass filter by zeroing out frequencies > cutoff_frequency
        SP500_fft_copy = np.copy(self.SP500_fft)
        SP500_fft_copy[self.cutoff_frequency:] = 0
        SP500_ifft_copy = np.fft.irfft(SP500_fft_copy).real

        plt.plot(self.SP500_open_val, label="Original Data", linestyle='--', color='blue')
        plt.plot(SP500_ifft_copy, label="Reconstructed Data", color='red')
        plt.xlabel('Business Day Number')
        plt.ylabel('S&P 500 Opening Value')
        plt.title('Comparison of Original and Reconstructed Data (Filtered)')
        plt.legend()
        plt.grid(True)
        plt.savefig("SP500_Og_Recon_63days.png")
        print("Saved: SP500_Og_Recon_63days.png")
        plt.clf()


def main():
    sp500_analysis = SP500Analysis('sp500.csv')
    sp500_analysis.plot_opening_values()
    sp500_analysis.plot_reconstructed_data()
    sp500_analysis.plot_filtered_reconstructed_data()


if __name__ == "__main__":
    main()
