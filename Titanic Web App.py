from turtle import width
import dash
from dash import html
import pandas as pd
import plotly.express as px
from dash import dcc
from dash.dependencies import Input, Output
from joblib import dump, load

#Juan Pablo Herrera, Santiago Rodas

df = pd.read_csv("test.csv")

nombre_de_las_variables_de_entrada = ['CryoSleep','RoomService']

planets = df["HomePlanet"].unique()

df_planets = df["HomePlanet"]

grouped = df.groupby(df.HomePlanet)
df_Earth = grouped.get_group("Earth")
df_Earth = df_Earth[nombre_de_las_variables_de_entrada].dropna()

df_Europa = grouped.get_group("Europa")
df_Europa = df_Europa[nombre_de_las_variables_de_entrada].dropna()

df_Mars = grouped.get_group("Mars")
df_Mars = df_Mars[nombre_de_las_variables_de_entrada].dropna()

counts = df["HomePlanet"].value_counts()

 

titanic_Predict = load("filename.joblib")

result_Earth = titanic_Predict.predict(df_Earth)
result_Europa = titanic_Predict.predict(df_Europa)
result_Mars = titanic_Predict.predict(df_Mars)

df_Earth_result = pd.DataFrame(result_Earth).value_counts()
df_Europa_result = pd.DataFrame(result_Europa).value_counts()
df_Mars_result = pd.DataFrame(result_Mars).value_counts()

#fig = px.imshow("Yotsuba.jpg")

app = dash.Dash(__name__)



app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Datos', children=[
           dash.dash_table.DataTable(df.to_dict('records'))
        ]),
        dcc.Tab(label='EDA', children=[
            html.H3(["Holi"]),
            dcc.Graph(
                figure={
                    'data': [
                        {'x': planets, 'y': [df_Earth_result[0],df_Europa_result[0],df_Mars_result[0]],
                            'type': 'bar', 'name': 'True'},
                        {'x': planets, 'y': [df_Earth_result[1],df_Europa_result[1],df_Mars_result[1]],
                            'type': 'bar', 'name': 'False'}
                    ], 'layout' :  {'title': 'Transportados'}
                }
            )
        ]),

        dcc.Tab(label='Quintillizas', children=[
            html.H3("En este programa se ama y respeta a las quintillizas"),
            dash.html.Img(src = '/assets/Sisters.png'),
            html.Div([ 
            html.Div([
             "CryoSleep: ",
                 dcc.Dropdown(['True', 'False'],
                     multi=False, id='my-input', optionHeight = 20),
                
             ], style={'padding': 10, 'flex': 1}),
            
            html.Div([
             "Room Service: ",
                dcc.Input(id='my-input2', value='initial value', type='number'),
                
             ], style={'padding': 10, 'flex': 1}),


             html.Div(id='my-output', style={'padding': 10, 'flex': 1})

             ], style={'display': 'flex', 'flex-direction': 'row'}),
            
        ]),
       
    ])
])

app.scripts.config.serve_locally = False

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
    Input(component_id='my-input2', component_property='value')
)
def update_output_div(cryoSleep, roomService):
    cryo = " "
    if cryoSleep == 'True' :

        cryo = True

        data = {'CryoSleep': [cryo], 'RoomService': [roomService]} 

        df_test = pd.DataFrame(data)  

        return 'Output: {}'.format(titanic_Predict.predict(df_test))

    elif cryoSleep == 'False':

        cryo = False

        data = {'CryoSleep': [cryo], 'RoomService': [roomService]} 

        df_test = pd.DataFrame(data)  
        
        return 'Output: {}'.format(titanic_Predict.predict(df_test))

    else : 

        return 'Output: {}'.format("Valor Incorrecto")

    



if __name__ == '__main__':
    app.run_server(debug=True)