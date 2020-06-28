# Input csv, get a plotly table as an html file (to include in github io)

# Feel free to make table with pandas dataframes


import plotly.graph_objects as go
import pandas as pd
from options import Options

if __name__ == "__main__":
    
    opt = Options().parse()
    df = pd.read_csv(opt.input_data_path)

    

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=(df.values).T,
                fill_color='lavender',
                align='left'))
    ])

    fig.show()

    fig.write_html(opt.output_html_pathname)