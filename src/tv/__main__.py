"""Entry point for the tv TIFF viewer."""

import numpy as np

from tv.viewer import ZoomableCfaViewer, load_tiff_file


def main():
    """Launch the TIFF viewer application."""
    # Try to load a TIFF file from file dialog
    my_data, filename = load_tiff_file()

    # If no file was loaded, use mock data
    if my_data is None:
        print("\nGenerating mock data for demonstration...")
        CFA_WIDTH = 2000
        CFA_HEIGHT = 1500
        print(f"Generating a mock {CFA_HEIGHT}x{CFA_WIDTH} 16-bit NumPy array...")
        # Create a gradient so it's not just noise
        x = np.linspace(0, 65535, CFA_WIDTH)
        y = np.linspace(0, 65535, CFA_HEIGHT)
        xx, yy = np.meshgrid(x, y)
        my_data = (xx + yy).astype(np.uint16)
        print("Mock data generated.")
        filename = None

    # Run the application
    print("Launching ZoomableCfaViewer...")
    app = ZoomableCfaViewer(cfa_data=my_data, filename=filename)
    app.mainloop()


if __name__ == "__main__":
    main()
