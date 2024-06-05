from matplotlib import cm
from matplotlib import colors           

import matplotlib.pyplot as plt       
import numpy as np


# =========================================================================================================================
class CVoronoi2DPlot(object): 
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_sTitle, p_nSamples2D, p_nLabels, p_nGroundTruthClusterCount=10):
    # ................................................................
    # // Fields \\
    self.Title                    = p_sTitle
    self.Samples2D                = p_nSamples2D
    self.Labels                   = p_nLabels
    self.GroundTruthClusterCount  = p_nGroundTruthClusterCount

    if (self.GroundTruthClusterCount <= 10):
      self.ColorMap   = cm.get_cmap("tab10")
    elif (self.GroundTruthClusterCount <= 20):
      self.ColorMap   = cm.get_cmap("tab20")
    else:
      self.ColorMap = cm.get_cmap("prism")

    self.PointSize = 8
    self.PlotDimensions = [14,8]
    # ................................................................
  # --------------------------------------------------------------------------------------
  def ShowForKMeans(self, p_oKMeansModel):
    reduced_data = self.Samples2D

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    #h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .4     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = p_oKMeansModel.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=self.PlotDimensions)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cm.get_cmap("tab20"), aspect="auto", origin="lower")

    #plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.Labels, s=self.PointSize, cmap=self.ColorMap )

    # Plot the centroids as a white X
    centroids = p_oKMeansModel.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3
                ,color="w", zorder=10 )
    plt.title(self.Title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
  # --------------------------------------------------------------------------------------