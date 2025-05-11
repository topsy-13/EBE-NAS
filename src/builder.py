import torch
import torch.nn as nn
import torch.optim as optim

import copy
import ast

class DynamicNN(nn.Module):  # MLP
    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 activation_fn, dropout_rate,
                 lr, optimizer_type, 
                 weight_decay=0, momentum=None,
                 use_skip_connections=False,
                 initializer='xavier_uniform', lr_scheduler='none',
                 scheduler_params={},
                 device=None):
        super(DynamicNN, self).__init__()

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_skip_connections = use_skip_connections
        
        layers = []
        prev_size = input_size

        for size in hidden_layers:
            layer = nn.Linear(prev_size, size)

            # Apply initializer
            if initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif initializer == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif initializer == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight)
            elif initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight)

            layers.append(layer)
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        # Get the optimizer class from string if needed
        optimizer_class = self._get_optimizer_class(optimizer_type)

        # Configure optimizer
        optimizer_kwargs = {'lr': lr}
        if weight_decay > 0:
            optimizer_kwargs['weight_decay'] = weight_decay
        
        # Handle momentum parameter
        if momentum is not None:
            # For SGD optimizer
            if optimizer_class == optim.SGD:
                optimizer_kwargs['momentum'] = momentum
            # For RMSprop optimizer which also supports momentum
            elif optimizer_class == optim.RMSprop:
                optimizer_kwargs['momentum'] = momentum

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # Scheduler
        if lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif lr_scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get('gamma', 0.9)
            )
        elif lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 10)
            )
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()
    
    def _get_optimizer_class(self, optimizer_type):
        """
        Convert optimizer_type to an actual optimizer class.
        
        Args:
            optimizer_type: Can be:
                - A direct reference to an optimizer class (e.g., optim.Adam)
                - A string representation of a class (e.g., "<class 'torch.optim.rmsprop.RMSprop'>")
                - A string name of an optimizer (e.g., "Adam", "SGD", "RMSprop")
        
        Returns:
            The optimizer class
        """
        # If optimizer_type is already a class, return it
        if isinstance(optimizer_type, type):
            return optimizer_type
        
        # If optimizer_type is a string representation of a class like "<class 'torch.optim.rmsprop.RMSprop'>"
        if isinstance(optimizer_type, str) and optimizer_type.startswith("<class '") and "'" in optimizer_type:
            try:
                # Extract the class path (handle both with and without closing bracket)
                if optimizer_type.endswith("'>"):
                    class_path = optimizer_type[8:-2]  # Remove "<class '" and "'>"
                else:
                    # Handle case where the closing bracket is missing
                    class_path = optimizer_type[8:].split("'")[0]
                
                # Split the path into components
                components = class_path.split('.')
                
                # Import the module and get the class
                module_path = '.'.join(components[:-1])  # e.g., 'torch.optim.rmsprop'
                class_name = components[-1]  # e.g., 'RMSprop'
                
                # Dynamically import the module
                module = __import__(module_path, fromlist=[class_name])
                
                # Get the class from the module
                return getattr(module, class_name)
            
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"Failed to parse optimizer class from '{optimizer_type}': {e}")
        
        # If optimizer_type is a simple string name of an optimizer
        if isinstance(optimizer_type, str):
            optimizer_map = {
                'sgd': optim.SGD,
                'adam': optim.Adam,
                'adamw': optim.AdamW,
                'rmsprop': optim.RMSprop,
                'adagrad': optim.Adagrad,
                'adadelta': optim.Adadelta
            }
            
            optimizer_key = optimizer_type.lower()
            if optimizer_key in optimizer_map:
                return optimizer_map[optimizer_key]
            
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        raise TypeError(f"optimizer_type must be a class, class string, or name string, got {type(optimizer_type)}")

    def forward(self, x):
        if not self.use_skip_connections:
            return self.network(x)

        result = x
        idx = 0

        for module in self.network:
            if isinstance(module, nn.Linear) and idx > 0:
                output = module(result)
                if result.shape == output.shape:
                    result = output + result
                else:
                    result = output
            else:
                result = module(result)
            idx += 1

        return result

        
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()


    def oe_train(self, train_loader, num_epochs=1):
        self.train()
        
        for epoch in range(num_epochs):
            total = 0
            correct = 0
            running_loss = 0.0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                if features.dim() > 2:  
                    features = features.view(features.size(0), -1)  # Flatten images

                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item() * features.size(0)

                # Compute accuracy
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Compute final loss and accuracy for the epoch
            train_loss = running_loss / total
            train_acc = correct / total
            
        return train_loss, train_acc
    
    def es_train(self, train_loader, val_loader, es_patience=50, max_epochs=300, verbose=False):
        best_val_acc = -float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        best_train_loss = None
        best_train_acc = None
        best_val_loss = None

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.oe_train(train_loader)

            self.eval()
            running_loss_val = 0.0
            total_val = 0
            correct_val = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)

                    outputs = self(features)
                    loss = self.criterion(outputs, labels)

                    running_loss_val += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = running_loss_val / total_val
            val_acc = correct_val / total_val

            self.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
                print('New best acc found:', best_val_acc)
            else:
                epochs_without_improvement += 1

            if verbose:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if epochs_without_improvement >= es_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch} epochs.")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return best_train_loss, best_train_acc, best_val_loss, best_val_acc

    def evaluate(self, val_loader):
            self.eval()
            
            total = 0
            correct = 0
            running_loss = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                    if features.dim() > 2:  
                        features = features.view(features.size(0), -1)

                    outputs = self(features)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * features.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = running_loss / total
            val_accuracy = correct / total
            
            return val_loss, val_accuracy


def create_model_from_row(row, input_size, output_size):

    # Hidden layers
    hidden_layers = row.get('hidden_layers', [128, 64])
    if isinstance(hidden_layers, str):
        hidden_layers = ast.literal_eval(hidden_layers)

    # Activation function
    activation_raw = row.get('activation_fn', nn.ReLU)
    if isinstance(activation_raw, str):
        activation_name = activation_raw
    elif hasattr(activation_raw, '__name__'):
        activation_name = activation_raw.__name__
    else:
        activation_name = 'ReLU'

    activation_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
    }
    activation_fn = activation_map.get(activation_name, nn.ReLU)

    # Dropout
    dropout_rate = row.get('dropout_rate', 0.2)

    # Optimizer and learning rate
    lr = row.get('lr', 0.001)
    optimizer_type = row.get('optimizer_type', 'adam')
    weight_decay = row.get('weight_decay', 0.0)
    momentum = row.get('momentum', None)

    # Skip connections
    use_skip = row.get('use_skip_connections', False)

    # Initializer
    initializer = row.get('initializer', 'xavier_uniform')

    # LR Scheduler
    lr_scheduler = row.get('lr_scheduler', 'none')
    scheduler_params = row.get('scheduler_params', {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = DynamicNN(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        lr=lr,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        momentum=momentum,
        use_skip_connections=use_skip,
        initializer=initializer,
        lr_scheduler=lr_scheduler,
        scheduler_params=scheduler_params,
        device=device
    ).to(device)

    return model