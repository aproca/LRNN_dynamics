import matplotlib.pyplot as plt
import matplotlib as mpl


# def plot_inout_mode(args, tape):
#     fig = plt.figure(dpi=300)
#     ax = fig.add_subplot(111)
#     indices = [0,1,3,2]
#     for alpha in indices:
#         sim_u = []
#         for i in range(len(W_x_ar)):
#             out_mode = (args.U.T @ W_y_ar[i] @ R_y)[alpha,alpha]
#             in_mode = (R_x.T @ W_x_ar[i] @ args.V)[alpha,alpha]
#             sim_u.append((in_mode*out_mode).detach().numpy())
#         theory_u_ar = []
#         for i in range(len(sim_u)):
#             theory_u_ar.append(compute_u_t(args, ))

#         # bsb2_ratio = 

#         ax.plot(sim_u, label=str(bsb2_ratio))
#         if alpha == 2:
#             ax.plot(theory_u_ar, ls='dashed', color='black', label='theory')
#         else:
#             ax.plot(theory_u_ar, ls='dashed', color='black')
#     ax.set_xlabel('t (epochs)')
#     ax.set_ylabel(r'input-output mode strength ($ac$)')
#     ax.legend(title=r'$\frac{\sum b_is_i}{\sum b_ib_i}$', frameon=False, bbox_to_anchor=(1,0.05))
#     plt.show()

class BlindColours: # consider looking at https://cmasher.readthedocs.io/user/diverging.html
    def __init__(self, reverse_cmap=True):
        hex_colours = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440", "#000000"]
        self.blind_colours = [mpl.colors.to_rgb(h) for h in hex_colours]

        # Define a diverging colormap
        div = ['#6d0000', '#720400', '#770900', '#7c0d00', '#821200', '#871600', '#8b1b00', '#901f00', '#952300',
               '#9a2700', '#9f2c00', '#a33000', '#a83400', '#ad3800', '#b13c00', '#b64000', '#bb4500', '#bf4900',
               '#c44d00', '#c85100', '#cc5604', '#cf5b09', '#d3600e', '#d66513', '#d96a18', '#dd6f1d', '#e07422',
               '#e37927', '#e67e2c', '#ea8331', '#ed8836', '#f08d3b', '#f3923f', '#f69744', '#f99b49', '#fda04e',
               '#ffa555', '#feac62', '#fdb26e', '#fdb87a', '#fcbe87', '#fbc492', '#faca9e', '#f9d5b4', '#f8dabf',
               '#f8e0ca', '#f7e5d5', '#f6ebe0', '#f6f0ea', '#ecf2f6', '#e3eef7', '#d9ebf8', '#d0e7f8', '#c6e4f9',
               '#bde0fa', '#b3ddfb', '#a9d9fc', '#9fd6fd', '#95d2fe', '#8bceff', '#85cafc', '#80c6f9', '#7bc2f6',
               '#75bef2', '#70baef', '#6bb6ec', '#66b1e9', '#61ade5', '#5ba9e2', '#56a5df', '#51a1dc', '#4c9dd8',
               '#4799d5', '#4295d2', '#3c91cf', '#378dcb', '#3289c8', '#2d85c5', '#2881c2', '#237dbf', '#2079ba',
               '#1e75b6', '#1c71b1', '#1a6dad', '#1969a8', '#1765a4', '#15619f', '#135d9b', '#115996', '#105592',
               '#0e518e', '#0c4d89', '#0a4a85', '#094681', '#07427d', '#053e79', '#033b74', '#023770', '#00346c']
        if reverse_cmap:
            div.reverse()  # Reverse the colormap if specified
        self.div_cmap = mpl.colors.ListedColormap(div)

        # Define color steps for specific color ranges
        oranges = [mpl.colors.to_rgb(h) for h in ['#871500', '#a93700', '#cc5400', '#ef721c', '#ff9c4a']]
        blues = [mpl.colors.to_rgb(h) for h in ['#00356e', '#005492', '#0975b7', '#4895d9', '#70b6fd']]
        greens = [mpl.colors.to_rgb(h) for h in ['#003e1d', '#005e39', '#008057', '#09a378', '#46c698']]
        self.colour_steps = [oranges, blues, greens]

    def get_colours(self):
        return self.blind_colours

    def get_div_cmap(self):
        return self.div_cmap

    def get_colour_steps(self):
        return self.colour_steps