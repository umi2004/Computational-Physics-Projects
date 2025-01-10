import numpy as np
import matplotlib.pyplot as plt

class SLPAnalysis:
    def __init__(self, slp_filename, lon_filename, times_filename):
        self.SLP = np.loadtxt(slp_filename)
        self.Longitude = np.loadtxt(lon_filename)
        self.Times = np.loadtxt(times_filename)
        self.SLP_fft = np.fft.fft(self.SLP, axis=1)

    def plot_wavenumber_components(self, wavenumbers, cmaps, output_filenames):
        for m, cmap, output_filename in zip(wavenumbers, cmaps, output_filenames):
            # Reconstruct and plot the component for given wavenumber m
            SLP_m = np.zeros_like(self.SLP_fft)
            SLP_m[:, m] = self.SLP_fft[:, m]
            SLP_m_recon = np.fft.ifft(SLP_m, axis=1).real
            plt.contourf(self.Longitude, self.Times, SLP_m_recon, cmap=cmap)
            plt.colorbar(label='SLP (hPa)')
            plt.title(f'SLP Component for Wavenumber m = {m}')
            plt.xlabel('Longitude (degrees)')
            plt.ylabel('Time (days)')
            plt.savefig(output_filename)
            print(f"Saved: {output_filename}")
            plt.close("all")

def main():
    # SLP Analysis
    slp_analysis = SLPAnalysis('SLP.txt', 'lon.txt', 'times.txt')
    wavenumbers = [3, 5]
    cmaps = ['GnBu', 'GnBu']
    output_filenames = ["SLP_3m.png", "SLP_5m.png"]
    slp_analysis.plot_wavenumber_components(wavenumbers, cmaps, output_filenames)

if __name__ == "__main__":
    main()
