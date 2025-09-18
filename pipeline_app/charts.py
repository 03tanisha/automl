import pandas as pd
import plotly.express as px

# Common theme function
def apply_dark_theme(fig, grid=True, width=300, height=250):
    fig.update_layout(
        plot_bgcolor="#222",   # chart area background
        paper_bgcolor="#222",  # full figure background
        font=dict(color="white"),  # text color
        xaxis=dict(showgrid=grid, gridcolor="#444"),
        yaxis=dict(showgrid=grid, gridcolor="#444"),
        width=width,
        height=height,
        margin=dict(l=40, r=40, t=60, b=40)  # keep charts neat, not crowded
    )
    return fig


def generate_charts(df, col):
    group = {"col": col, "charts": []}

    if pd.api.types.is_numeric_dtype(df[col]):
        # Histogram
        fig_hist = px.histogram(df, x=col, nbins=18, title=f"Histogram of {col}")
        fig_hist.update_traces(marker_color="#599e94", hovertemplate="%{x}:%{y}")
        fig_hist = apply_dark_theme(fig_hist, grid=True)
        group["charts"].append(fig_hist.to_html(full_html=False))

        # Boxplot
        fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
        fig_box.update_traces(marker_color="#08bdba", hovertemplate="%{y}")
        fig_box = apply_dark_theme(fig_box, grid=False)
        group["charts"].append(fig_box.to_html(full_html=False))

    else:
        # Bar chart
        top_values = df[col].value_counts().nlargest(5)
        fig_bar = px.bar(x=top_values.index, y=top_values.values, title=f"Top 5 values in {col}")
        fig_bar.update_traces(marker_color="#599e94", hovertemplate="%{x}: %{y}")
        fig_bar = apply_dark_theme(fig_bar, grid=True)
        group["charts"].append(fig_bar.to_html(full_html=False))

        # Pie chart
        fig_pie = px.pie(values=top_values.values, names=top_values.index, title=f"Pie chart of {col}")
        fig_pie.update_traces(hoverinfo='label+percent', textinfo='value')
        fig_pie = apply_dark_theme(fig_pie, grid=False)
        group["charts"].append(fig_pie.to_html(full_html=False))

    return group
