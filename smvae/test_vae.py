import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for astronomical photometry data"""
    
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=2):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with optional beta-VAE weighting"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class SkyMapperDataLoader:
    """Class to handle SkyMapper DR4 data retrieval and preprocessing"""
    
    def __init__(self):
        self.vizier = Vizier(columns=['*'], row_limit=-1)
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
    def query_skymapper_dr4(self, ra_center=180.0, dec_center=0.0, radius=1.0, 
                           g_mag_limit=14.0, sample_size=10000):
        """
        Query SkyMapper DR4 for stars brighter than g_mag_limit
        
        Parameters:
        -----------
        ra_center : float
            Right ascension center in degrees
        dec_center : float 
            Declination center in degrees
        radius : float
            Search radius in degrees
        g_mag_limit : float
            Magnitude limit in g band
        sample_size : int
            Maximum number of stars to retrieve
        """
        
        print(f"Querying SkyMapper DR4 around RA={ra_center}, Dec={dec_center}")
        print(f"Search radius: {radius} degrees")
        print(f"g-band magnitude limit: {g_mag_limit}")
        
        try:
            # Define the catalog
            catalog = "II/358/smss"  # SkyMapper Southern Sky Survey DR4
            
            # Create coordinate for cone search
            coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
            
            # Query the catalog
            result = self.vizier.query_region(
                coord, 
                radius=radius*u.deg, 
                catalog=catalog
            )
            
            if len(result) == 0:
                print("No data found in the specified region")
                return None
                
            # Get the table
            table = result[0]
            print(f"Initial query returned {len(table)} objects")
            
            # Convert to pandas DataFrame
            df = table.to_pandas()
            
            # Filter by g-band magnitude
            if 'gmag' in df.columns:
                df = df[df['gmag'] < g_mag_limit]
            elif 'g_psf' in df.columns:
                df = df[df['g_psf'] < g_mag_limit]
            else:
                print("Warning: g-band magnitude column not found")
                
            print(f"After g-band filtering: {len(df)} objects")
            
            # Sample if too many objects
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"Sampled down to {sample_size} objects")
                
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error querying SkyMapper: {e}")
            print("Generating synthetic data for demonstration...")
            return self.generate_synthetic_data(sample_size)
    
    def generate_synthetic_data(self, n_samples=10000):
        """Generate realistic synthetic photometric data for demonstration"""
        np.random.seed(42)
        
        # Generate realistic stellar colors based on stellar populations
        # Main sequence stars
        ms_colors = {
            'g-i': np.random.normal(0.5, 0.3, int(n_samples * 0.7)),
            'v-g': np.random.normal(-0.1, 0.2, int(n_samples * 0.7)),
            'r-i': np.random.normal(0.2, 0.15, int(n_samples * 0.7)),
            'i-z': np.random.normal(0.1, 0.1, int(n_samples * 0.7))
        }
        
        # Giant stars
        giant_colors = {
            'g-i': np.random.normal(1.2, 0.4, int(n_samples * 0.2)),
            'v-g': np.random.normal(0.3, 0.3, int(n_samples * 0.2)),
            'r-i': np.random.normal(0.4, 0.2, int(n_samples * 0.2)),
            'i-z': np.random.normal(0.25, 0.15, int(n_samples * 0.2))
        }
        
        # White dwarfs
        wd_colors = {
            'g-i': np.random.normal(-0.2, 0.1, int(n_samples * 0.1)),
            'v-g': np.random.normal(-0.3, 0.1, int(n_samples * 0.1)),
            'r-i': np.random.normal(-0.1, 0.05, int(n_samples * 0.1)),
            'i-z': np.random.normal(0.0, 0.05, int(n_samples * 0.1))
        }
        
        # Combine all populations
        colors = {}
        for color_name in ['g-i', 'v-g', 'r-i', 'i-z']:
            colors[color_name] = np.concatenate([
                ms_colors[color_name],
                giant_colors[color_name], 
                wd_colors[color_name]
            ])
        
        # Add some correlations between colors (realistic for stellar populations)
        colors['v-g'] += 0.3 * colors['g-i'] + np.random.normal(0, 0.1, n_samples)
        colors['r-i'] += 0.5 * colors['g-i'] + np.random.normal(0, 0.05, n_samples)
        colors['i-z'] += 0.2 * colors['g-i'] + np.random.normal(0, 0.03, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(colors)
        
        # Add some additional columns for realism
        df['gmag'] = np.random.uniform(10, 14, n_samples)
        df['ra'] = np.random.uniform(175, 185, n_samples)
        df['dec'] = np.random.uniform(-5, 5, n_samples)
        
        print(f"Generated {n_samples} synthetic stellar photometry samples")
        self.data = df
        return df
    
    def preprocess_data(self):
        """Preprocess the photometric data for VAE training"""
        if self.data is None:
            print("No data available. Please query data first.")
            return None
            
        # Check for available magnitude columns
        mag_columns = ['gPSF', 'iPSF', 'vPSF', 'rPSF', 'zPSF']
        available_mags = [col for col in mag_columns if col in self.data.columns]
        
        if len(available_mags) < 4:
            print(f"Not enough magnitude bands available. Found: {available_mags}")
            print("Available columns:", self.data.columns.tolist())
            return None
            
        # Create a copy for processing
        work_data = self.data.copy()
        
        # Remove rows with NaN values in magnitude columns
        initial_count = len(work_data)
        work_data = work_data.dropna(subset=available_mags)
        print(f"Removed {initial_count - len(work_data)} rows with NaN magnitude values")
        
        # Compute color indices
        color_data = pd.DataFrame()
        
        if 'gPSF' in work_data.columns and 'iPSF' in work_data.columns:
            color_data['g-i'] = work_data['gPSF'] - work_data['iPSF']
            
        if 'vPSF' in work_data.columns and 'gPSF' in work_data.columns:
            color_data['v-g'] = work_data['vPSF'] - work_data['gPSF']
            
        if 'rPSF' in work_data.columns and 'iPSF' in work_data.columns:
            color_data['r-i'] = work_data['rPSF'] - work_data['iPSF']
            
        if 'iPSF' in work_data.columns and 'zPSF' in work_data.columns:
            color_data['i-z'] = work_data['iPSF'] - work_data['zPSF']
            
        # If we don't have enough colors, fall back to using magnitude differences
        if len(color_data.columns) < 4:
            print("Computing alternative color indices...")
            color_data = pd.DataFrame()
            
            # Use any available magnitude differences
            if 'gPSF' in work_data.columns and 'rPSF' in work_data.columns:
                color_data['g-r'] = work_data['gPSF'] - work_data['rPSF']
            if 'gPSF' in work_data.columns and 'iPSF' in work_data.columns:
                color_data['g-i'] = work_data['gPSF'] - work_data['iPSF']
            if 'rPSF' in work_data.columns and 'iPSF' in work_data.columns:
                color_data['r-i'] = work_data['rPSF'] - work_data['iPSF']
            if 'iPSF' in work_data.columns and 'zPSF' in work_data.columns:
                color_data['i-z'] = work_data['iPSF'] - work_data['zPSF']
                
        # Filter out obvious g-band sources if available
        if 'gPSF' in work_data.columns:
            g_mask = work_data['gPSF'] < 14.0
            color_data = color_data[g_mask]
            print(f"Applied g-band magnitude filter: {len(color_data)} stars remain")
        
        # Remove rows with NaN values in color columns
        initial_count = len(color_data)
        color_data = color_data.dropna()
        print(f"Removed {initial_count - len(color_data)} rows with NaN color values")
        
        if len(color_data) == 0:
            print("No valid color data remaining. Exiting.")
            return None
            
        # Remove outliers (beyond 5 sigma)
        for col in color_data.columns:
            mean_val = color_data[col].mean()
            std_val = color_data[col].std()
            outlier_mask = np.abs(color_data[col] - mean_val) > 5 * std_val
            color_data = color_data[~outlier_mask]
            
        print(f"Final dataset size: {len(color_data)} stars")
        print(f"Color indices computed: {list(color_data.columns)}")
        
        if len(color_data) < 1000:
            print("Warning: Very small dataset for VAE training")
            
        # Standardize the data
        scaled_data = self.scaler.fit_transform(color_data)
        
        self.processed_data = scaled_data
        self.color_names = list(color_data.columns)
        
        return scaled_data

def train_vae(data, epochs=100, batch_size=256, learning_rate=1e-3, 
              latent_dim=2, hidden_dim=64, beta=1.0):
    """Train the VAE model"""
    
    # Convert to PyTorch tensors
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = data.shape[1]
    model = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    recon_losses = []
    kl_losses = []
    
    print(f"Training VAE for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch in dataloader:
            batch_data = batch[0]
            
            # Forward pass
            recon_batch, mu, logvar, z = model(batch_data)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
        # Store losses
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
        avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, "
                  f"Recon = {avg_recon_loss:.4f}, KL = {avg_kl_loss:.4f}")
    
    return model, losses, recon_losses, kl_losses

def explore_latent_space(model, data, scaler, color_names, n_samples=2000):
    """Explore and visualize the learned latent space"""
    
    model.eval()
    
    # Sample data for visualization
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data
    
    # Encode to latent space
    with torch.no_grad():
        tensor_data = torch.FloatTensor(sample_data)
        mu, logvar = model.encode(tensor_data)
        z = model.reparameterize(mu, logvar)
        z_numpy = z.numpy()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Latent space scatter plot
    ax = axes[0, 0]
    scatter = ax.scatter(z_numpy[:, 0], z_numpy[:, 1], 
                        c=sample_data[:, 0], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(f'Latent Space (colored by {color_names[0]})')
    plt.colorbar(scatter, ax=ax)
    
    # 2. Latent space density
    ax = axes[0, 1]
    ax.hist2d(z_numpy[:, 0], z_numpy[:, 1], bins=50, cmap='Blues')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Latent Space Density')
    
    # 3. Reconstruction quality
    ax = axes[0, 2]
    with torch.no_grad():
        recon_data, _, _, _ = model(tensor_data)
        recon_error = torch.mean((tensor_data - recon_data)**2, dim=1).numpy()
    
    ax.scatter(z_numpy[:, 0], z_numpy[:, 1], c=recon_error, 
              cmap='Reds', alpha=0.6, s=20)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Reconstruction Error')
    
    # 4. Color-color plots in original space
    # Inverse transform to get original colors
    original_colors = scaler.inverse_transform(sample_data)
    
    ax = axes[1, 0]
    ax.scatter(original_colors[:, 0], original_colors[:, 1], 
              c=z_numpy[:, 0], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel(color_names[0])
    ax.set_ylabel(color_names[1])
    ax.set_title(f'{color_names[0]} vs {color_names[1]} (colored by latent dim 1)')
    
    ax = axes[1, 1]
    ax.scatter(original_colors[:, 2], original_colors[:, 3], 
              c=z_numpy[:, 1], cmap='plasma', alpha=0.6, s=20)
    ax.set_xlabel(color_names[2])
    ax.set_ylabel(color_names[3])
    ax.set_title(f'{color_names[2]} vs {color_names[3]} (colored by latent dim 2)')
    
    # 5. Latent space interpolation
    ax = axes[1, 2]
    
    # Generate a grid in latent space
    n_grid = 20
    z1_range = np.linspace(z_numpy[:, 0].min(), z_numpy[:, 0].max(), n_grid)
    z2_range = np.linspace(z_numpy[:, 1].min(), z_numpy[:, 1].max(), n_grid)
    z1_grid, z2_grid = np.meshgrid(z1_range, z2_range)
    
    # Decode grid points
    grid_points = np.column_stack([z1_grid.ravel(), z2_grid.ravel()])
    with torch.no_grad():
        decoded_grid = model.decode(torch.FloatTensor(grid_points))
        decoded_numpy = decoded_grid.numpy()
    
    # Show the first color component
    decoded_reshaped = decoded_numpy[:, 0].reshape(n_grid, n_grid)
    im = ax.imshow(decoded_reshaped, extent=[z1_range.min(), z1_range.max(), 
                                           z2_range.min(), z2_range.max()],
                   cmap='viridis', origin='lower')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(f'Decoded {color_names[0]} in Latent Space')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return z_numpy

def main():
    """Main function to run the complete analysis"""
    
    print("=== SkyMapper DR4 Variational Autoencoder Analysis ===\n")
    
    # 1. Load and preprocess data
    print("1. Loading SkyMapper DR4 data...")
    loader = SkyMapperDataLoader()
    
    # Query data (using synthetic data for demonstration)
    df = loader.query_skymapper_dr4(
        ra_center=180.0, 
        dec_center=-30.0, 
        radius=2.0,
        g_mag_limit=14.0,
        sample_size=15000
    )
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"\nData summary:")
    print(f"Number of stars: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    processed_data = loader.preprocess_data()
    
    if processed_data is None:
        print("Failed to preprocess data. Exiting.")
        return
    
    print(f"Processed data shape: {processed_data.shape}")
    
    # 3. Train VAE
    print("\n3. Training Variational Autoencoder...")
    model, losses, recon_losses, kl_losses = train_vae(
        processed_data,
        epochs=150,
        batch_size=256,
        learning_rate=1e-3,
        latent_dim=2,
        hidden_dim=128,
        beta=1.0
    )
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses)
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Explore latent space
    print("\n4. Exploring latent space...")
    latent_coords = explore_latent_space(
        model, processed_data, loader.scaler, loader.color_names
    )
    
    # 5. Additional analysis
    print("\n5. Additional analysis...")
    
    # Compute latent space statistics
    print(f"Latent space statistics:")
    print(f"Dimension 1: mean = {latent_coords[:, 0].mean():.3f}, "
          f"std = {latent_coords[:, 0].std():.3f}")
    print(f"Dimension 2: mean = {latent_coords[:, 1].mean():.3f}, "
          f"std = {latent_coords[:, 1].std():.3f}")
    
    # Sample new stars from the latent space
    print("\n6. Generating new stellar colors from latent space...")
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z_sample = torch.randn(10, 2)
        generated_colors = model.decode(z_sample)
        generated_colors_numpy = generated_colors.numpy()
        
        # Inverse transform to original scale
        original_generated = loader.scaler.inverse_transform(generated_colors_numpy)
        
        print("Generated stellar colors:")
        for i, colors in enumerate(original_generated):
            print(f"Star {i+1}: g-i={colors[0]:.3f}, v-g={colors[1]:.3f}, "
                  f"r-i={colors[2]:.3f}, i-z={colors[3]:.3f}")
    
    print("\n=== Analysis Complete ===")
    
    return model, loader, latent_coords

if __name__ == "__main__":
    # Run the complete analysis
    try:
        model, loader, latent_coords = main()
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Using synthetic data for demonstration...")
        
        # Fallback to synthetic data
        loader = SkyMapperDataLoader()
        df = loader.generate_synthetic_data(15000)
        processed_data = loader.preprocess_data()
        
        if processed_data is not None:
            print("\nTraining VAE with synthetic data...")
            model, losses, recon_losses, kl_losses = train_vae(
                processed_data, epochs=100, batch_size=256, learning_rate=1e-3,
                latent_dim=2, hidden_dim=128, beta=1.0
            )
            
            # Plot training losses
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(losses)
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(recon_losses)
            plt.title('Reconstruction Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 3)
            plt.plot(kl_losses)
            plt.title('KL Divergence Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.tight_layout()
            plt.show()
            
            # Explore latent space
            latent_coords = explore_latent_space(
                model, processed_data, loader.scaler, loader.color_names
            )
            
            print("Analysis completed successfully with synthetic data!")

