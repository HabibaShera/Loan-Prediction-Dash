import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class HandleReturns:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data = self.data.dropna()
        self.layout = go.Layout(plot_bgcolor = "rgba(255,255,255,0)", paper_bgcolor = "rgba(255,255,255,0)")
        #self.layout.update_layout(coloraxis=dict(colorscale='Viridis'))
        
    
    def return_pie(self, col_name):
        data = self.data[col_name].value_counts()
        fig = go.Figure(layout=self.layout, layout_showlegend=False , data=[go.Pie(labels= data.index, values=data.values, textinfo='label+percent', textfont_size=17)])
        #fig = px.pie(self.data, values=values, names=names, title=title)
        fig.update_layout(title=f'{col_name.title()}', title_font=dict(size=26, color='white'),
                      title_x=0.5, title_y=0.95,)
        return fig
    
    def return_hist(self, x, title):
        fig = go.Figure(layout=self.layout, data=[go.Histogram(x=self.data[x])])
        fig.update_xaxes(tickfont=dict(size=17))
        fig.update_yaxes(tickfont=dict(size=17))
        fig.update_traces(marker=dict(color='white'))
        fig.update_layout(title=f'{x.title()}  Distribution', title_font=dict(size=26, color='white'),
                      title_x=0.5, title_y=0.95,)
        #if color:
        #    fig = px.histogram(self.data, x=x, y=y, color=color, barmode='group')
        #else:
        #    fig = px.histogram(self.data, x=x, y=y, title=title)     
        return fig       
    
    def return_grouped_bar(self, x, y):
        grouped_df = self.groupby(col1=x, col2=y)
        fig = go.Figure(layout=self.layout, data=[go.Bar(
            x=grouped_df[x][grouped_df[y] == cat2],
            y=grouped_df['Loan_ID'][grouped_df[y] == cat2],
            name=cat2)
        for cat2 in grouped_df[y].unique()
        ])        #fig =  px.bar(self.data, x=x, y=y, color=color, barmode='group', layout = layout)
        
        
        fig.update_layout(title=f'{y.title()}  vs Loan Status', title_font=dict(size=26, color='white'),
                      title_x=0.5, title_y=0.95,)
        return fig
    
    def groupby(self, col1, col2):
        grouped_df = self.data.groupby([col1, col2]).count().reset_index()
        return grouped_df
    


