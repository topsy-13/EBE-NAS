import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc

# region Generations
class Generation():
    def __init__(self, search_space, n_individuals):
        self.search_space = search_space
        self.n_individuals = n_individuals
        self.generation = self.build_generation() 

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture()
            model = self.search_space.create_model(architecture)
            generation[i] = {
                "model": model,
                "architecture": architecture,
                "batch_size": architecture['batch_size']
            }
        return generation
    
    def train_generation(self, X_train, y_train, num_epochs=1):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)

            train_loss, train_acc = model.oe_train(train_loader, num_epochs=num_epochs)
            self.generation[i]["train_loss"] = train_loss
            self.generation[i]["train_acc"] = train_acc
    

    def validate_generation(self, X_val, y_val):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            
            # Create a DataLoader with the architecture-specific batch size
            val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
            
            val_loss, val_acc = model.evaluate(val_loader)
            self.generation[i]["val_loss"] = val_loss
            self.generation[i]["val_acc"] = val_acc


    def get_worst_individuals(self, 
                              percentile_drop=15):
    
        n_worst_individuals = max(1, int(self.n_individuals * percentile_drop / 100))  # Ensure at least 1

        # Sort individuals by validation loss in descending order (higher loss is worse)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"], reverse=True) #? Should the criterion be val_loss or val_acc?

        # Extract the keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]

        

    def drop_worst_individuals(self):
        # Clean up GPU memory before removing references
        for idx in self.worst_individuals:
            if hasattr(self.generation[idx]["model"], "cpu"):
                self.generation[idx]["model"] = self.generation[idx]["model"].cpu()
            # Force garbage collection for the model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.generation[idx]
        
        # Re-index the remaining individuals to maintain continuous keys
        self.generation = {new_idx: val for new_idx, (_, val) in enumerate(self.generation.items())}
        self.n_individuals = len(self.generation)  # Update the count

    def drop_all_except_best(self):
        # Sort individuals by validation loss in ascending order (lower loss is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"])
        
        # Keep only the best individual
        best_individual = sorted_generation[0][0]
        best_model_data = self.generation[best_individual]
        
        # Clean up GPU memory for models that will be discarded
        for idx, data in self.generation.items():
            if idx != best_individual:
                if hasattr(data["model"], "cpu"):
                    data["model"] = data["model"].cpu()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.generation = {0: best_model_data}
        self.n_individuals = 1

    def train_best_individual(self, X_train, y_train, num_epochs=1):
        best_model = self.generation[0]["model"]
        batch_size = self.generation[0]["batch_size"]
        # Create a DataLoader with the architecture-specific batch size
        train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
        best_model.oe_train(train_loader, num_epochs=num_epochs)
    
    def return_df(self):
        # As a dataframe
        architectures = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        batch_sizes = []
        for i in range(self.n_individuals):
            architectures.append(self.generation[i]["architecture"])
            train_losses.append(self.generation[i]["train_loss"])
            train_accs.append(self.generation[i]["train_acc"])
            val_losses.append(self.generation[i]["val_loss"])
            val_accs.append(self.generation[i]["val_acc"])
            batch_sizes.append(self.generation[i]["batch_size"])        

        # Create a DataFrame with the architectures and their corresponding metrics
        architectures_df = pd.DataFrame(architectures)
        architectures_df['train_loss'] = train_losses
        architectures_df['train_acc'] = train_accs
        architectures_df['val_loss'] = val_losses
        architectures_df['val_acc'] = val_accs
        architectures_df['batch_size'] = batch_sizes

        df = pd.DataFrame(architectures_df)
        return df.sort_values('val_acc', ascending=False).reset_index(drop=True)

    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25):
    
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train, num_epochs=1)
        self.validate_generation(X_val, y_val)
        self.get_worst_individuals(percentile_drop)
        self.drop_worst_individuals()

        return self.generation
    
    def run_ebe(self, n_epochs,
                X_train, y_train, X_val, y_val,
                percentile_drop=25):
        
        for n_epoch in range(n_epochs):
            print(f"Epoch {n_epoch+1}/{n_epochs}")
            self.generation = self.run_generation(X_train, y_train, X_val, y_val, percentile_drop=percentile_drop)
            self.num_models = len(self.generation)
            percentile_drop += 5


# region Functions

def create_dataloaders(X, y, 
                       batch_size, return_as='loaders'):

    # Create DataLoaders
    dataset, dataloader = load_data.create_dataset_and_loader(X, y,
    batch_size=batch_size)
    if return_as == 'loaders':
        return dataloader
    else: 
        return dataset