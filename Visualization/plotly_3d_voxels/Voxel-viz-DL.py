# Get this figure: fig = py.get_figure("https://plotly.com/~empet/15251/")
# Get this figure's data: data = py.get_figure("https://plotly.com/~empet/15251/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="Voxel-viz-DL", fileopt="extend")
# Get y data of first trace: y1 = py.get_figure("https://plotly.com/~empet/15251/").get_data()[0]["y"]

# Get figure documentation: https://plotly.com/python/get-requests/
# Add data documentation: https://plotly.com/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plotly.com/python/getting-started
# Find your api_key here: https://plotly.com/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
# py.sign_in('username', 'api_key')
trace1 = {
  "isrc": "empet:15261:eabea9", 
  "i": [0], 
  "jsrc": "empet:15261:43b165",  
  "j": [181], 
  "ksrc": "empet:15261:3ea764", 
  "k": [100],
  "meta": {"columnNames": {
      "i": "data.0.i", 
      "j": "data.0.j", 
      "k": "data.0.k", 
      "x": "data.0.x", 
      "y": "data.0.y", 
      "z": "data.0.z"
    }}, 
  "type": "mesh3d", 
  "xsrc": "empet:15261:64cbbd", 
  "x": [0.0], 
  "ysrc": "empet:15261:fea0db", 
  "y": [0.0], 
  "zsrc": "empet:15261:58775e", 
  "z": [0.0], 
  "color": "#ce6a6b", 
  "lighting": {
    "ambient": 0.5, 
    "diffuse": 1, 
    "fresnel": 4, 
    "specular": 0.5, 
    "roughness": 0.5
  }, 
  "flatshading": True
}
data = Data([trace1])
layout = {
  "scene": {
    "xaxis": {
      "type": "linear", 
      "ticks": "", 
      "title": {"text": ""}, 
      "showticklabels": False
    }, 
    "yaxis": {
      "type": "linear", 
      "ticks": "", 
      "title": {"text": ""}, 
      "showticklabels": False
    }, 
    "zaxis": {
      "type": "linear", 
      "ticks": "", 
      "title": {"text": ""}, 
      "showticklabels": False
    }, 
    "camera": {
      "up": {
        "x": 0, 
        "y": 0, 
        "z": 1
      }, 
      "eye": {
        "x": 0.7045312826318519, 
        "y": -1.7713815474172667, 
        "z": 0.5552864893572735
      }, 
      "center": {
        "x": 0, 
        "y": 0, 
        "z": 0
      }, 
      "projection": {"type": "perspective"}
    }, 
    "aspectmode": "data", 
    "aspectratio": {
      "x": 0.8461970835132063, 
      "y": 0.7791681509661961, 
      "z": 1.5166916076699701
    }
  }, 
  "title": {
    "x": 0.5, 
    "text": "Voxel  visualization for deep learning of 3d data"
  }, 
  "height": 600, 
  "template": {
    "data": {
      "bar": [
        {
          "type": "bar", 
          "marker": {"line": {
              "color": "#E5ECF6", 
              "width": 0.5
            }}, 
          "error_x": {"color": "#2a3f5f"}, 
          "error_y": {"color": "#2a3f5f"}
        }
      ], 
      "table": [
        {
          "type": "table", 
          "cells": {
            "fill": {"color": "#EBF0F8"}, 
            "line": {"color": "white"}
          }, 
          "header": {
            "fill": {"color": "#C8D4E3"}, 
            "line": {"color": "white"}
          }
        }
      ], 
      "carpet": [
        {
          "type": "carpet", 
          "aaxis": {
            "gridcolor": "white", 
            "linecolor": "white", 
            "endlinecolor": "#2a3f5f", 
            "minorgridcolor": "white", 
            "startlinecolor": "#2a3f5f"
          }, 
          "baxis": {
            "gridcolor": "white", 
            "linecolor": "white", 
            "endlinecolor": "#2a3f5f", 
            "minorgridcolor": "white", 
            "startlinecolor": "#2a3f5f"
          }
        }
      ], 
      "mesh3d": [
        {
          "type": "mesh3d", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "contour": [
        {
          "type": "contour", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ], 
      "heatmap": [
        {
          "type": "heatmap", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ], 
      "scatter": [
        {
          "type": "scatter", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "surface": [
        {
          "type": "surface", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ], 
      "barpolar": [
        {
          "type": "barpolar", 
          "marker": {"line": {
              "color": "#E5ECF6", 
              "width": 0.5
            }}
        }
      ], 
      "heatmapgl": [
        {
          "type": "heatmapgl", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ], 
      "histogram": [
        {
          "type": "histogram", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "parcoords": [
        {
          "line": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}, 
          "type": "parcoords"
        }
      ], 
      "scatter3d": [
        {
          "line": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}, 
          "type": "scatter3d", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scattergl": [
        {
          "type": "scattergl", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "choropleth": [
        {
          "type": "choropleth", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "scattergeo": [
        {
          "type": "scattergeo", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "histogram2d": [
        {
          "type": "histogram2d", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ], 
      "scatterpolar": [
        {
          "type": "scatterpolar", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "contourcarpet": [
        {
          "type": "contourcarpet", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "scattercarpet": [
        {
          "type": "scattercarpet", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scattermapbox": [
        {
          "type": "scattermapbox", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scatterpolargl": [
        {
          "type": "scatterpolargl", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scatterternary": [
        {
          "type": "scatterternary", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "histogram2dcontour": [
        {
          "type": "histogram2dcontour", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
          ]
        }
      ]
    }, 
    "layout": {
      "geo": {
        "bgcolor": "white", 
        "showland": True, 
        "lakecolor": "white", 
        "landcolor": "#E5ECF6", 
        "showlakes": True, 
        "subunitcolor": "white"
      }, 
      "font": {"color": "#2a3f5f"}, 
      "polar": {
        "bgcolor": "#E5ECF6", 
        "radialaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "angularaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }
      }, 
      "scene": {
        "xaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }, 
        "yaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }, 
        "zaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }
      }, 
      "title": {"x": 0.05}, 
      "xaxis": {
        "ticks": "", 
        "gridcolor": "white", 
        "linecolor": "white", 
        "automargin": True, 
        "zerolinecolor": "white", 
        "zerolinewidth": 2
      }, 
      "yaxis": {
        "ticks": "", 
        "gridcolor": "white", 
        "linecolor": "white", 
        "automargin": True, 
        "zerolinecolor": "white", 
        "zerolinewidth": 2
      }, 
      "mapbox": {"style": "light"}, 
      "ternary": {
        "aaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "baxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "caxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "bgcolor": "#E5ECF6"
      }, 
      "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], 
      "hovermode": "closest", 
      "colorscale": {
        "diverging": [
          [0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]
        ],
        "sequential": [
          [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
        ],
        "sequentialminus": [
          [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921"]
        ]
      }, 
      "hoverlabel": {"align": "left"}, 
      "plot_bgcolor": "#E5ECF6", 
      "paper_bgcolor": "white", 
      "shapedefaults": {"line": {"color": "#2a3f5f"}}, 
      "annotationdefaults": {
        "arrowhead": 0, 
        "arrowcolor": "#2a3f5f", 
        "arrowwidth": 1
      }
    }
  }
}
fig.show
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)