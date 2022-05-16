from tkinter import HIDDEN
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from joblib import dump, load
from sklearn.datasets import load_digits
import pylab as pl
import random
from dash.dependencies import Input, Output



#Juan Pablo Herrera y Santiago Rodas
app = dash.Dash(__name__)

Mnist = load("MnistCL.joblib")

df = load_digits()

rando = random.randint(0,len(df.images))
fig = px.imshow(df.images[rando])

print("random value ", rando)

app.layout = html.Div(children=[
    html.Div(id='data', hidden= HIDDEN, children= rando),
    html.Div(id='data_output', hidden= HIDDEN, children= rando),
    html.Button('Generate Random Number', id='submit-val', n_clicks=0),
    dcc.Graph(id="graph-pic", figure=fig ),
    html.Button('Mostrar Prediccion', id='submit-pred', n_clicks=0),
    html.Div(id='prediction_output'),
    
    
])


@app.callback(
    Output(component_id='graph-pic', component_property='figure'),
    Output(component_id='data_output', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    Input(component_id='data', component_property='children')
)
def update_output_div(randy,number): 
    rando = random.randint(0,len(df.images))
    number = rando
    fig = px.imshow(df.images[rando])
    return fig, number


@app.callback(
    Output(component_id='prediction_output', component_property='children'),
    Input(component_id='submit-pred', component_property='n_clicks'),
    Input(component_id='data_output', component_property='children')
)
def predict_output_div(randy,number):
    prediction = Mnist.predict(df.images[number].reshape(1,-1))
    lorezo = prediction[0]
    result = "El modelo predijo: ", lorezo
    return  result




if __name__ == '__main__':
    app.run_server(debug=True)