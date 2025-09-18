import pandas as pd 
import plotly.express as px

def corr_chart(df):
    num_cols = df.select_dtypes(include=['int64','float64'])
    if num_cols.empty:
        return"<p>No numeric column available for the correlation matrix</p>"
    corr = num_cols.corr()
    # sns.heatmap(corr, annot=True, cmap='coolwarm')
    fig=px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', aspect='auto',)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor='rgba(0,0,0,1)',   # dark background
                      plot_bgcolor='rgba(0,0,0,1)',
                      font_color='white',)
    fig_html = fig.to_html(full_html=False)

    return fig_html



    
