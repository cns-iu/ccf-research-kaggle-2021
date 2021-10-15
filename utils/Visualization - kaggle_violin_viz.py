import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import pandas as pd

target_list = ['glom', 'crypt']
target = 0
target_root_path = rf"violin_data/{target_list[target]}"
team_names = ['1-tom', '2-gleb', '3-wgo', '4-dl', '5-df2']

color_dict = {'1-tom-glom': 'black',
              '1-tom-slide': 'grey',
              '2-gleb-glom': 'orangered',
              '2-gleb-slide': 'orange',
              '3-wgo-glom': 'midnightblue',
              '3-wgo-slide': 'royalblue',
              '4-dl-glom': 'darkolivegreen',
              '4-dl-slide': 'mediumseagreen',
              '5-df2-glom': 'purple',
              '5-df2-slide': 'violet',
              }

point_position_dict = {'1-tom-slide': 0.8,
                       '1-tom-glom': -1.1,
                       '2-gleb-slide': 0.8,
                       '2-gleb-glom': -1.1,
                       '3-wgo-slide': 0.8,
                       '3-wgo-glom': -1.1,
                       '4-dl-slide': 0.8,
                       '4-dl-glom': -1.1,
                       '5-df2-slide': 0.8,
                       '5-df2-glom': -1.1,
                       }

opacity_dict = {'glom': 0.7,
                'slide': 0.7}
legend_dict = {'glom': {'glom': 'Glomeruli level',
                        'slide': 'Slide level'},
               'crypt': {'glom': 'Crypt level',
                         'slide': 'Slide level'}
               }
location_dict = {'dice': [1, 1],
                 'recall': [1, 2],
                 'precision': [2, 2],
                 }

data_types = ['dice', 'recall', 'precision']
glom_data_list = {
    'dice': {},
    'recall': {},
    'precision': {},
}
slide_data_list = {
    'dice': {},
    'recall': {},
    'precision': {},
}

for team in team_names:
    file_path = target_root_path + rf"\{team}.txt"
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        glom_data_list['dice'][team] = [float(line.split(',')[0]) for line in lines if len(line) > 1]
        glom_data_list['recall'][team] = [float(line.split(',')[1]) for line in lines if len(line) > 1]
        glom_data_list['precision'][team] = [float(line.split(',')[2]) for line in lines if len(line) > 1]

    slide_file_path = target_root_path + rf"\{team}_sum.txt"
    with open(slide_file_path, 'r') as f:
        lines = f.read().splitlines()
        slide_data_list['dice'][team] = [float(line.split(',')[0]) for line in lines if len(line) > 1]
        slide_data_list['recall'][team] = [float(line.split(',')[1]) for line in lines if len(line) > 1]
        slide_data_list['precision'][team] = [float(line.split(',')[2]) for line in lines if len(line) > 1]

# display box and scatter plot along with violin plot
# fig = pt.violin(n_data, y="distance", x="Age", color="Skin Type",
#                 box=True, hover_data=n_data.columns,
#                 points='all',
#                 )
# fig = go.Figure()

fig = make_subplots(
    rows=2, cols=2,
    row_heights=[0.5, 0.5],
    # specs=[[{"type": "Scatter3d", "colspan": 4}, None, None, None],
    #        [{"type": "Histogram"}, {"type": "Histogram"}, {"type": "Histogram"}, {"type": "Histogram"}]],
    shared_xaxes=True,
    vertical_spacing=0.04,
    horizontal_spacing=0.04,
    # subplot_titles=[f'All', 'CD68 / Macrophage', 'T-Helper', 'T-Regulatory'],
    subplot_titles=[f'Dice coefficient', 'Recall', 'Precision', ],
    specs=[[{"secondary_y": False, "rowspan": 2}, {"secondary_y": False}],
           [None, {"secondary_y": False}], ],
)

# fig.add_trace(go.Violin(x=n_data['Region'][n_data['Skin Type'] == 'Sun-Exposed'],
#                         y=n_data['distance'][n_data['Skin Type'] == 'Sun-Exposed'],
#                         name='Sun-Exposed', legendgroup='Sun-Exposed',
#                         line_color='orange', points="outliers",
#                         box_visible=True, width=2,
#                         meanline_visible=True))
# fig.add_trace(go.Violin(x=n_data['Region'][n_data['Skin Type'] == 'Non-Sun-Exposed'],
#                         y=n_data['distance'][n_data['Skin Type'] == 'Non-Sun-Exposed'],
#                         name='Non-Sun-Exposed', legendgroup='Non-Sun-Exposed',
#                         line_color='blue', points="outliers",
#                         box_visible=True, width=2,
#                         meanline_visible=True))


# region_seq = [12, 5, 4, 2, 10, 7, 11, 3, 6, 8, 9, 1, ]
# for index in range(len(regions)):
#     next = region_seq[index] - 1
#     fig.add_trace(go.Violin(x=n_data['Age'][n_data['Region'] == str(regions[next])],
#                             y=n_data['distance'][n_data['Region'] == str(regions[next])],
#                             name=suns[next], legendgroup=suns[next],
#                             line_color=color_dict[suns[next]], points="outliers",
#                             box_visible=True, width=1,
#                             meanline_visible=True))
# for skin_type in ['Sun-Exposed', 'Non-Sun-Exposed']:
#     fig.add_trace(go.Violin(x=n_data['Age'][n_data['Skin Type'] == skin_type],
#                             y=n_data['distance'][n_data['Skin Type'] == skin_type],
#                             name=skin_type, legendgroup='All', legendgrouptitle_text="All",
#                             points="outliers", opacity=opacity_dict[skin_type], width=4,
#                             box_visible=True, line_color=color_dict[skin_type], meanline_visible=False),
#                   secondary_y=False, row=1, col=1, )

for data_type in data_types:
    for team in team_names:
        for level_type in ['glom', 'slide']:
            fig.add_trace(
                go.Violin(x=[team] * (len(glom_data_list[data_type][team])
                                      if level_type == 'glom' else len(slide_data_list[data_type][team])),
                          y=(glom_data_list[data_type][team]
                             if level_type == 'glom' else slide_data_list[data_type][team]),
                          name=team,
                          points="all", opacity=opacity_dict[level_type],
                          pointpos=point_position_dict[f'{team}-{level_type}'],
                          side=('positive' if level_type == 'slide' else 'negative'),
                          legendgroup=level_type, showlegend=True if data_type == 'dice' else False,
                          scalemode='width', scalegroup="all", width=0,  # level_type + data_type,
                          jitter=0.05, marker_opacity=0.5, marker_size=2, line_width=1, spanmode='soft',
                          legendgrouptitle_text=legend_dict[target_list[target]][level_type],
                          box_visible=True, box_fillcolor='white',
                          line_color=color_dict[f'{team}-{level_type}'], meanline_visible=True, ),
                secondary_y=False, row=location_dict[data_type][0], col=location_dict[data_type][1],
            )

# fig.update_traces(# meanline_visible=False,
#                   scalemode='count')  # scale violin plot area with total count
title_texts = ["Kidney - dice/recall/precision  [glom level (~2000 matching gloms) \n/ slide level (10 slides)]",
               "Colon - dice/recall/precision  [crypt level (~160 matching crypts) \n/ slide level (2 slides)]"]
fig.update_layout(
    title=title_texts[target],
    # x1axis_title="Age",
    # yaxis_title="Dice",
    violingap=0, violingroupgap=0,
    violinmode='overlay',
    yaxis_zeroline=False,
    font=dict(
        family="Bahnschrift, Arial",
        size=16,
        # color="RebeccaPurple"
    ))
# sub plot title font size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=22)

fig.update_yaxes(tickvals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
fig.update_yaxes(tickfont=dict(size=14), col=2)

fig.write_html(os.path.join(target_root_path, f"kaggle_{target_list[target]}_violin.html"))
fig.show()
