import torch


class Regressor(torch.nn.Module):
    """
    Defines the networks f and g, where y = f(x) and flattened_matrix = g(x)
    """
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int) -> None:
        super(Regressor,self).__init__()
        
        activation = torch.nn.ELU
        mu_blocks = 5
        var_blocks = 5

        # Network to predict the mean
        self.mu_model = [torch.nn.Linear(in_dim, latent_dim), activation()]
        self.mu_model.extend(
            [torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim), activation(),
                torch.nn.Linear(latent_dim, latent_dim), torch.nn.BatchNorm1d(latent_dim),
                activation())
            for _ in range(mu_blocks)])
        self.mu_model.append(torch.nn.Linear(latent_dim, out_dim))

        self.mu_length = len(self.mu_model)
        self.mu_model = torch.nn.ModuleList(self.mu_model)

        # Network to predict the covariance or precision
        self.var_model = [torch.nn.Linear(in_dim, latent_dim), activation()]
        self.var_model.extend(
            [torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim), activation(),
                torch.nn.Linear(latent_dim, latent_dim), torch.nn.BatchNorm1d(latent_dim),
                activation())
            for _ in range(var_blocks)])
        self.var_model.append(torch.nn.Linear(latent_dim, (out_dim ** 2) + 2))

        self.var_length = len(self.var_model)
        self.var_model = torch.nn.ModuleList(self.var_model)


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
      
        x_mu = x.clone()
        x_var = x.clone()

        x_mu = self.mu_model[0](x_mu)
        for i in range(1, self.mu_length - 1):
            x_mu = x_mu + self.mu_model[i](x_mu)
        x_mu = self.mu_model[-1](x_mu)

        x_var = self.var_model[0](x_var)
        for i in range(1, self.var_length - 1):
            x_var = x_var + self.var_model[i](x_var)
        x_var = self.var_model[-1](x_var)
        
        return x_mu, x_var