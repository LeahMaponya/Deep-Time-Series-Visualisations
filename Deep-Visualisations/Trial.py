import numpy
import pandas as pd
import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from sklearn import datasets
import matplotlib.pyplot as plt



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seed the random number generator for reproducibility
rng = np.random.default_rng(123)

# Build a dataset with 10 prediction points and 4 features for each
df = pd.DataFrame({
    "name": [f"Prediction {i}" for i in range(1, 11) for _ in range(4)],
    "value": rng.integers(low=30, high=100, size=40),
    "feature": ["Feature 1", "Feature 2", "Feature 3", "Feature 4"] * 10
})

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 4

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label in zip(angles, values, labels):
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )

# Calculate angles and setup for the plot
PAD = 2  # Reduced padding for a more compact plot
GROUPS_SIZE = [len(i[1]) for i in df.groupby("name")]
ANGLES_N = len(df) + PAD * len(GROUPS_SIZE)
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / ANGLES_N

# Initialize IDXS to determine the positions of bars
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Assign colors to the bars based on their feature
COLORS = [f"C{i%4}" for i in range(len(df))]
VALUES = df["value"].values
LABELS = df["feature"].values

# Plotting
fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={"projection": "polar"})

OFFSET = np.pi / 2
ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Draw bars
ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
    edgecolor="white", linewidth=2
)

# Add labels
add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Display the plot
plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
data = pd.DataFrame({
    'individual': [f'Mister {i}' for i in range(1, 61)],
    'group': ['A'] * 10 + ['B'] * 30 + ['C'] * 14 + ['D'] * 6,
    'value1': np.random.randint(10, 101, 60),
    'value2': np.random.randint(10, 101, 60),
    'value3': np.random.randint(10, 101, 60)
})

data = pd.melt(data, id_vars=['individual', 'group'], var_name='observation', value_name='value')

empty_bar = 2
nObsType = data['observation'].nunique()
to_add = pd.DataFrame(np.nan, index=range(empty_bar * data['group'].nunique() * nObsType), columns=data.columns)
to_add['group'] = np.repeat(data['group'].unique(), empty_bar * nObsType)
data = pd.concat([data, to_add])
data = data.sort_values(['group', 'individual'])
data['id'] = np.repeat(range(1, len(data) // nObsType + 1), nObsType)

label_data = data.groupby(['id', 'individual'])['value'].sum().reset_index(name='tot')
number_of_bar = len(label_data)

base_data = data.groupby('group').agg({'id': ['min', 'max']}).reset_index()
base_data.columns = ['group', 'start', 'end']
base_data['end'] = base_data['end'] - empty_bar

grid_data = base_data.copy()
grid_data['end'] = np.roll(grid_data['end'] + 1, 1)
grid_data['start'] = grid_data['start'] - 1
grid_data = grid_data.iloc[1:]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for obs in data['observation'].unique():
    obs_data = data[data['observation'] == obs]
    ax.bar(obs_data['id'] * 2 * np.pi / number_of_bar, obs_data['value'],
           width=2*np.pi/number_of_bar, bottom=0, alpha=0.5)

for _, row in grid_data.iterrows():
    ax.plot([row['start']*2*np.pi/number_of_bar, row['end']*2*np.pi/number_of_bar],
            [0, 0], color='grey', alpha=0.3)
    for y in [50, 100, 150, 200]:
        ax.plot([row['start']*2*np.pi/number_of_bar, row['end']*2*np.pi/number_of_bar],
                [y, y], color='grey', alpha=0.3)

ax.set_ylim(-150, label_data['tot'].max())

ax.set_yticks([])
ax.set_xticks([])

for _, row in base_data.iterrows():
    start_angle = row['start'] * 2 * np.pi / number_of_bar
    end_angle = row['end'] * 2 * np.pi / number_of_bar
    ax.plot([start_angle, end_angle], [-5, -5], color='black', alpha=0.8, linewidth=0.6)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
data = pd.DataFrame({
    'individual': [f'Mister {i}' for i in range(1, 61)],
    'group': ['A'] * 10 + ['B'] * 30 + ['C'] * 14 + ['D'] * 6,
    'value1': np.random.randint(10, 101, 60),
    'value2': np.random.randint(10, 101, 60),
    'value3': np.random.randint(10, 101, 60)
})

data = pd.melt(data, id_vars=['individual', 'group'], var_name='observation', value_name='value')

empty_bar = 2
nObsType = data['observation'].nunique()
to_add = pd.DataFrame(np.nan, index=range(empty_bar * data['group'].nunique() * nObsType), columns=data.columns)
to_add['group'] = np.repeat(data['group'].unique(), empty_bar * nObsType)
data = pd.concat([data, to_add])
data = data.sort_values(['group', 'individual'])
data['id'] = np.repeat(range(1, len(data) // nObsType + 1), nObsType)

label_data = data.groupby(['id', 'individual'])['value'].sum().reset_index(name='tot')
number_of_bar = len(label_data)

base_data = data.groupby('group').agg({'id': ['min', 'max']}).reset_index()
base_data.columns = ['group', 'start', 'end']
base_data['end'] = base_data['end'] - empty_bar

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Adjust bar width to create spaces
bar_width = 0.8 * 2 * np.pi / number_of_bar

for obs in data['observation'].unique():
    obs_data = data[data['observation'] == obs]
    ax.bar(obs_data['id'] * 2 * np.pi / number_of_bar,
           obs_data['value'],
           width=bar_width,
           bottom=0,
           alpha=0.5)

ax.set_ylim(-150, label_data['tot'].max())
ax.set_yticks([])
ax.set_xticks([])

for _, row in base_data.iterrows():
    start_angle = row['start'] * 2 * np.pi / number_of_bar
    end_angle = row['end'] * 2 * np.pi / number_of_bar
    ax.plot([start_angle, end_angle], [-5, -5], color='black', alpha=0.8, linewidth=0.6)

# Keep the circular spine visible
ax.spines['polar'].set_visible(True)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
path1 ='data.pickle'
with open(path1, 'rb') as f:
    features = pickle.load(f)

print(features)
features_1= features[0]
print(features_1)


file_path = '/home/leah/Downloads/Deeplearn_data_global.pkl'
path = '/home/leah/projects/KnowIt/leahs-synth-demo/leahs_mlp/interpretations/DeepLiftShap-eval-success-100-True-(2923, 3023).pickle'
d='/home/leah/PycharmProjects/Starter/data.pickle'
# Open the pickle file for reading
with open(path, 'rb') as file:
    # Load the data from the pickle file
    res = pickle.load(file)
print(res)
feat_att1=res['results'][2923][(0,0)]['attributions']
feat_att2=res['results'][2924][(0,0)]['attributions']
feat_att3=res['results'][2925][(0,0)]['attributions']
feat_att4=res['results'][2926][(0,0)]['attributions']
feat_att5=res['results'][2927][(0,0)]['attributions']
feat_att6=res['results'][2928][(0,0)]['attributions']


import matplotlib.pyplot as plt
import numpy as np

# Assuming you extracted feature attributions correctly
feat_atts = [res['results'][i][(0, 0)]['attributions'].detach().numpy() for i in range(2923, 2933)]  # Adjust range for number of predictions

# Number of prediction points, features, and time steps (adjusted based on your data)
prediction_points = len(feat_atts)
features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
time_steps = 4  # Adjusted to match the size of the last dimension of the data

# Convert the list of arrays into a 3D NumPy array for easy plotting
data = np.array(feat_atts)

# Create a plot
fig, ax = plt.subplots(figsize=(12, 6))

# Positioning of bars on the x-axis for prediction points
x = np.arange(prediction_points)

# Bar width and space between feature groups within each prediction point
bar_width = 0.15
space_between_features = 0.05

# Colors for the different time steps in each segment
colors = plt.cm.viridis(np.linspace(0, 1, time_steps))

# Plotting stacked bars for each feature within each prediction point
for i, feature in enumerate(features):
    for j in range(time_steps):
        if j == 0:
            ax.bar(x + i * (bar_width + space_between_features), data[:, i, j], bar_width, label=f'Time Step {j+1}', color=colors[j])
        else:
            ax.bar(x + i * (bar_width + space_between_features), data[:, i, j], bar_width, bottom=np.sum(data[:, i, :j], axis=1), color=colors[j])

# Labeling
ax.set_xlabel('Prediction Points')
ax.set_ylabel('Feature Attribution')
ax.set_title('Stacked Bar Plot of Feature Attributions for Each Prediction Point')
ax.set_xticks(x + (bar_width + space_between_features) * 1.5)  # Adjusting x-axis labels for clarity
ax.set_xticklabels([ i+1 for i in range(prediction_points)])

# Adjusting the legend to show each time step color
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:time_steps], labels[:time_steps], loc='upper right', title='Time Steps')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming feat_atts is defined as described

# Create dataset from feature attributions
data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': chr(97 + i),  # a, b, c, d, e...
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step,
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
n_time_steps = 11
width = 2 * np.pi / (n_predictions * n_features + n_predictions)  # added space between prediction groups

# Define colors for time steps
colors = plt.cm.viridis(np.linspace(0, 1, n_time_steps))

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angles = [i * (n_features + 1) * width + j * width + width / 2]

        bottom = 0
        for time_step, value in enumerate(feature_data['value']):
            ax.bar(angles, [abs(value)], width=width, bottom=bottom,
                   color=colors[time_step], alpha=0.7)
            bottom += abs(value)

# Set ylim and remove yticks
max_value = data.groupby(['prediction', 'feature'])['value'].sum().max()
ax.set_ylim(0, max_value * 1.1)
ax.set_yticks([])

# Add prediction labels
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * (n_features + 1) * width + (n_features / 2) * width
    ax.text(angle, max_value * 1.15, prediction, ha='center', va='center',
            fontsize=20, fontweight='bold')

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1, fontsize=24)

# Add legend for time steps
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_time_steps)]
ax.legend(legend_elements, [f'Time {i + 1}' for i in range(n_time_steps)],
          loc='center', bbox_to_anchor=(0.5, -0.1), ncol=6)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming feat_atts is defined as described

# Create dataset from feature attributions
data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': chr(97 + i),  # a, b, c, d, e...
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step,
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
n_time_steps = 11
width = 2 * np.pi / (n_predictions * n_features + n_predictions * 2)  # added more space between prediction groups

# Define colors for time steps
colors = plt.cm.viridis(np.linspace(0, 1, n_time_steps))

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angles = [i * (n_features + 2) * width + j * width + width / 2]

        bottom_pos = 0
        bottom_neg = 0
        for time_step, value in enumerate(feature_data['value']):
            if value >= 0:
                ax.bar(angles, [value], width=width, bottom=bottom_pos,
                       color=colors[time_step], alpha=0.7)
                bottom_pos += value
            else:
                ax.bar(angles, [abs(value)], width=width, bottom=bottom_neg,
                       color=colors[time_step], alpha=0.7)
                bottom_neg -= abs(value)

# Set ylim and remove yticks
max_value = max(data.groupby(['prediction', 'feature'])['value'].sum().max(),
                abs(data.groupby(['prediction', 'feature'])['value'].sum().min()))
ax.set_ylim(-max_value * 1.1, max_value * 1.1)
ax.set_yticks([])

# Add prediction labels inside the circle
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * (n_features + 2) * width + (n_features / 2) * width
    ax.text(angle, 0, prediction, ha='center', va='center',
            fontsize=20, fontweight='bold')

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1, fontsize=24)

# Add legend for time steps
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_time_steps)]
ax.legend(legend_elements, [f'Time {i + 1}' for i in range(n_time_steps)],
          loc='center', bbox_to_anchor=(0.5, -0.1), ncol=6)

# Add circular structure in the middle
circle = plt.Circle((0, 0), 0.2, transform=ax.transData._b, color="white", zorder=3)
ax.add_artist(circle)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming feat_atts is defined as described

# Create dataset from feature attributions
data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': chr(97 + i),  # a, b, c, d, e...
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step - 5,  # Adjust time steps to match legend
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
n_time_steps = 11
width = 2 * np.pi / (n_predictions * n_features + n_predictions * 2)  # added more space between prediction groups

# Define colors for time steps
colors = plt.cm.RdYlGn(np.linspace(0, 1, n_time_steps))

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angles = [i * (n_features + 2) * width + j * width + width / 2]

        bottom_pos = 0
        bottom_neg = 0
        for _, row in feature_data.iterrows():
            value = row['value']
            if value >= 0:
                ax.bar(angles, [value], width=width, bottom=bottom_pos,
                       color=colors[row['time_step'] + 5], alpha=0.7)
                bottom_pos += value
            else:
                ax.bar(angles, [abs(value)], width=width, bottom=bottom_neg,
                       color=colors[row['time_step'] + 5], alpha=0.7)
                bottom_neg -= abs(value)

# Set ylim and remove yticks
max_value = max(abs(data.groupby(['prediction', 'feature'])['value'].sum().max()),
                abs(data.groupby(['prediction', 'feature'])['value'].sum().min()))
ax.set_ylim(-max_value * 1.1, max_value * 1.1)
ax.set_yticks([])

# Add prediction labels inside the circle
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * (n_features + 2) * width + (n_features / 2) * width
    ax.text(angle, 0, prediction, ha='center', va='center',
            fontsize=20, fontweight='bold')

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1, fontsize=24)

# Add legend for time steps
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_time_steps)]
ax.legend(legend_elements, [f'Time {i - 5}' for i in range(n_time_steps)],
          loc='center', bbox_to_anchor=(1.2, 0.5), title="Time")

# Add circular structure in the middle
circle = plt.Circle((0, 0), 0.2, transform=ax.transData._b, color="white", zorder=3)
ax.add_artist(circle)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming feat_atts is defined as described

# Create dataset from feature attributions
data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': chr(97 + i),  # a, b, c, d, e...
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step - 5,  # Adjust time steps to match legend
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
n_time_steps = 11
width = 2 * np.pi / (n_predictions * n_features + n_predictions * 2)  # added more space between prediction groups

# Define colors for time steps
colors = plt.cm.RdYlGn(np.linspace(0, 1, n_time_steps))

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angle = i * (n_features + 2) * width + j * width + width / 2

        bottom_pos = 0
        bottom_neg = 0
        for _, row in feature_data.iterrows():
            value = row['value']
            if value >= 0:
                ax.bar([angle], [-value], width=width, bottom=bottom_pos,
                       color=colors[row['time_step'] + 5], alpha=0.7)
                bottom_pos -= value
            else:
                ax.bar([angle], [abs(value)], width=width, bottom=bottom_neg,
                       color=colors[row['time_step'] + 5], alpha=0.7)
                bottom_neg += abs(value)

# Set ylim and remove yticks
max_value = max(abs(data.groupby(['prediction', 'feature'])['value'].sum().max()),
                abs(data.groupby(['prediction', 'feature'])['value'].sum().min()))
ax.set_ylim(-max_value * 1.1, max_value * 1.1)
ax.set_yticks([])

# Add prediction labels inside the circle
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * (n_features + 2) * width + (n_features / 2) * width
    ax.text(angle, 0, prediction, ha='center', va='center',
            fontsize=20, fontweight='bold')

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1, fontsize=24)

# Add legend for time steps
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_time_steps)]
ax.legend(legend_elements, [f'Time {i - 5}' for i in range(n_time_steps)],
          loc='center', bbox_to_anchor=(1.2, 0.5), title="Time")

# Add circular structure in the middle
circle = plt.Circle((0, 0), 0.2, transform=ax.transData._b, color="white", zorder=3)
ax.add_artist(circle)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming feat_atts is defined as described

# Create dataset from feature attributions
data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': chr(97 + i),  # a, b, c, d, e...
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step,
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
width = 2 * np.pi / (n_predictions * n_features)

# Define colors for features
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angles = [i * n_features * width + j * width]

        bottom = 0
        for _, row in feature_data.iterrows():
            value = abs(row['value'])  # Use absolute value for all bars
            ax.bar(angles, [value], width=width, bottom=bottom,
                   color=colors[j], alpha=0.7)
            bottom += value

# Set ylim and remove yticks
max_value = data.groupby(['prediction', 'feature'])['value'].sum().abs().max()
ax.set_ylim(0, max_value * 1.1)
ax.set_yticks([])

# Add prediction labels inside the circle
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * n_features * width + (n_features / 2) * width
    ax.text(angle, 0.1, prediction, ha='center', va='center',
            fontsize=20, fontweight='bold')

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1, fontsize=24)

# Add legend for features
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
ax.legend(legend_elements, [f'Feature {i + 1}' for i in range(4)],
          loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# Add circular structure in the middle
circle = plt.Circle((0, 0), 0.2, transform=ax.transData._b, color="white", zorder=3)
ax.add_artist(circle)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assuming you extracted feature attributions correctly
feat_atts = [res['results'][i][(0, 0)]['attributions'].detach().numpy() for i in range(2923, 2933)]

# Number of prediction points, features, and time steps (adjusted based on your data)
prediction_points = len(feat_atts)
features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
time_steps = feat_atts[0].shape[0]  # Extracted from the shape of the feature attributions

# Convert the list of arrays into a 3D NumPy array for easy plotting
data = np.array(feat_atts)

# Create a plot
fig, ax = plt.subplots(figsize=(12, 6))

# Positioning of bars on the x-axis for prediction points
x = np.arange(prediction_points)

# Bar width and space between feature groups within each prediction point
bar_width = 0.15
space_between_features = 0.05

# Colors for the different time steps in each segment
colors = plt.cm.viridis(np.linspace(0, 1, time_steps))

# Create a list to store handles for the legend
legend_handles = []

# Plotting stacked bars for each feature within each prediction point
for i, feature in enumerate(features):
    for j in range(time_steps):
        if j == 0:
            # Create a bar plot for the first time step
            bar = ax.bar(x + i * (bar_width + space_between_features), data[:, j, i], bar_width,
                         label=f'Time Step {j+1}', color=colors[j])
            # Add the first time step bar to the legend handles
            legend_handles.append(bar)
        else:
            # Plot stacked bars with bottom parameter for subsequent time steps
            bottom_values = np.sum(data[:, :j, i], axis=1)
            ax.bar(x + i * (bar_width + space_between_features), data[:, j, i], bar_width,
                   bottom=bottom_values, color=colors[j])

# Labeling
ax.set_xlabel('Prediction Points')
ax.set_ylabel('Feature Attribution')
ax.set_title('Stacked Bar Plot of Feature Attributions for Each Prediction Point')
ax.set_xticks(x + (bar_width + space_between_features) * (len(features) / 2))  # Adjusting x-axis labels for clarity
ax.set_xticklabels([f'Pred {i+1}' for i in range(prediction_points)])

# Adjusting the legend to show each time step color
ax.legend(handles=legend_handles, loc='upper right', title='Time Steps')

plt.show()

import numpy as np

# Assuming you have your original data in a list called 'feat_atts'
feat_atts = [res['results'][i][(0, 0)]['attributions'].detach().numpy() for i in range(2923, 2933)]

# Convert the list of 11x4 arrays into a single 110x4 array
reshaped_data = np.vstack(feat_atts)

print(reshaped_data.shape)  # This should output (110, 4)

import numpy as np
import matplotlib.pyplot as plt

# Assuming reshaped_data is your 110x4 numpy array

def classify_value(x):
    if x > 0:
        return 0
    elif -0.05 < x <= 0:
        return 1
    elif -0.15 < x <= -0.05:
        return 2
    else:
        return 3

# Define the ranges
ranges = ['x > 0', '0 > x > -0.05', '-0.05 > x > -0.15', 'x < -0.15']

# Calculate the counts for each feature and range
counts = np.zeros((4, 4))
for feature in range(4):
    for value in reshaped_data[:, feature]:
        counts[feature, classify_value(value)] += 1

# Create the bubble chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each range
colors = ['#FFA07A', '#98FB98', '#87CEFA', '#DDA0DD']

for feature in range(4):
    for range_idx in range(4):
        count = counts[feature, range_idx]
        if count > 0:  # Only plot if there are values in this range
            ax.scatter(feature, range_idx, s=count*10, alpha=0.6, color=colors[range_idx])
            ax.annotate(f'{int(count)}\n{ranges[range_idx]}',
                        (feature, range_idx),
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center')

ax.set_xlabel('Features')
ax.set_ylabel('Value Ranges')
ax.set_xticks(range(4))
ax.set_xticklabels(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
ax.set_yticks(range(4))
ax.set_yticklabels(ranges)
ax.set_title('Feature Attribution Clusters')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming reshaped_data is your 110x4 numpy array

def classify_value(x):
    if x > 0:
        return 0
    elif -0.05 < x <= 0:
        return 1
    elif -0.15 < x <= -0.05:
        return 2
    else:
        return 3

# Define the ranges (we'll use these for the y-axis labels)
ranges = ['x > 0', '0 > x > -0.05', '-0.05 > x > -0.15', 'x < -0.15']

# Calculate the counts for each feature and range
counts = np.zeros((4, 4))
for feature in range(4):
    for value in reshaped_data[:, feature]:
        counts[feature, classify_value(value)] += 1

# Create the bubble chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each range
colors = ['#FFA07A', '#98FB98', '#87CEFA', '#DDA0DD']

for feature in range(4):
    for range_idx in range(4):
        count = counts[feature, range_idx]
        if count > 0:  # Only plot if there are values in this range
            ax.scatter(feature, range_idx, s=count*10, alpha=0.6, color=colors[range_idx])
            ax.annotate(f'{int(count)}',
                        (feature, range_idx),
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center')

ax.set_xlabel('Features')
ax.set_ylabel('Value Ranges')
ax.set_xticks(range(4))
ax.set_xticklabels(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
ax.set_yticks(range(4))
ax.set_yticklabels(ranges)
ax.set_title('Feature Attribution Clusters')

plt.tight_layout()
plt.show()


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming you have your reshaped data in 'reshaped_data' (110x4 numpy array)

# Number of clusters
n_clusters = 3  # You can adjust this number as needed

# Perform k-means clustering for each feature
kmeans_results = []

for feature in range(4):
    # Extract the data for this feature
    feature_data = reshaped_data[:, feature].reshape(-1, 1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_data)

    # Store the results
    kmeans_results.append({
        'kmeans': kmeans,
        'labels': kmeans.labels_,
        'centroids': kmeans.cluster_centers_
    })

# Visualize the results
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for feature in range(4):
    # Get the results for this feature
    results = kmeans_results[feature]

    # Plot the data points colored by cluster
    axs[feature].scatter(range(110), reshaped_data[:, feature], c=results['labels'], cmap='viridis')

    # Plot the centroids
    for centroid in results['centroids']:
        axs[feature].axhline(y=centroid, color='r', linestyle='--')

    axs[feature].set_title(f'Feature {feature + 1} Clustering')
    axs[feature].set_xlabel('Time Step')
    axs[feature].set_ylabel('Feature Value')

plt.tight_layout()
plt.show()

# Print cluster centroids
for feature in range(4):
    print(f"\nFeature {feature + 1} cluster centroids:")
    for i, centroid in enumerate(kmeans_results[feature]['centroids']):
        print(f"  Cluster {i + 1}: {centroid[0]:.4f}")

data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': f'Pred {i + 2923}',
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step,
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(projection='polar'))

# Calculate positions
n_predictions = len(feat_atts)
n_features = 4
n_time_steps = 11
bar_width = 2 * np.pi / (n_predictions * n_features * n_time_steps)

# Define colors for features
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        angles = np.linspace(i * 2 * np.pi / n_predictions,
                             (i + 1) * 2 * np.pi / n_predictions,
                             n_time_steps, endpoint=False)

        radii = feature_data['value'].abs().values

        ax.bar(angles, radii, width=bar_width, bottom=j * 0.5,
               color=colors[j], alpha=0.7, align='edge')

# Set ylim and remove yticks
ax.set_ylim(0, 2.5)
ax.set_yticks([])

# Add prediction labels
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * 2 * np.pi / n_predictions + np.pi / n_predictions
    ax.text(angle, 2.7, prediction, ha='center', va='center',
            rotation=np.degrees(angle) - 90)

# Remove xticks and add circular gridlines
ax.set_xticks([])
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1)

# Add legend
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7)
                   for color in colors]
ax.legend(legend_elements, ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
          loc='center', bbox_to_anchor=(0.5, -0.1), ncol=4)

plt.tight_layout()
plt.show()

data = []
for i, feat_att in enumerate(feat_atts):
    for feat_idx in range(4):
        for time_step in range(11):
            data.append({
                'prediction': f'Prediction {i + 2923}',
                'feature': f'Feature {feat_idx + 1}',
                'time_step': time_step,
                'value': feat_att[time_step, feat_idx]
            })

data = pd.DataFrame(data)

# Calculate the number of bars and set up the plot
number_of_predictions = len(feat_atts)
number_of_features = 4
number_of_time_steps = 11
total_bars = number_of_predictions * number_of_features * number_of_time_steps

fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(projection='polar'))

# Calculate bar positions and widths
bar_width = 2 * np.pi / total_bars
positions = np.linspace(0, 2 * np.pi, total_bars, endpoint=False)

# Plot the bars
for i, prediction in enumerate(data['prediction'].unique()):
    prediction_data = data[data['prediction'] == prediction]
    for j, feature in enumerate(prediction_data['feature'].unique()):
        feature_data = prediction_data[prediction_data['feature'] == feature]
        feature_data = feature_data.sort_values('time_step')

        start_index = i * number_of_features * number_of_time_steps + j * number_of_time_steps
        end_index = start_index + number_of_time_steps

        radii = feature_data['value'].abs().values
        colors = plt.cm.viridis(j / number_of_features)

        ax.bar(positions[start_index:end_index], radii, width=bar_width, bottom=0, alpha=0.7, color=colors)

# Remove yticks and set ylim
ax.set_yticks([])
ax.set_ylim(0, data['value'].abs().max() * 1.1)

# Add labels for predictions
for i, prediction in enumerate(data['prediction'].unique()):
    angle = i * number_of_features * number_of_time_steps * bar_width + (
                number_of_features * number_of_time_steps * bar_width) / 2
    ax.text(angle, ax.get_ylim()[1] * 1.1, prediction, ha='center', va='center', rotation=np.degrees(angle) - 90)

# Remove xticks
ax.set_xticks([])

# Add a title
plt.title("Feature Attributions for Multiple Predictions", y=1.1)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Assuming you extracted feature attributions correctly
feat_atts = [res['results'][i][(0, 0)]['attributions'].detach().numpy() for i in range(2923, 2933)]

# Number of prediction points, features, and time steps (adjusted based on your data)
prediction_points = len(feat_atts)
features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
time_steps = feat_atts[0].shape[0]  # Number of time steps from the shape of the feature attributions

# Convert the list of arrays into a 3D NumPy array for easy plotting
data = np.array(feat_atts)  # Shape: (prediction_points, time_steps, features)

# Create a plot
fig, ax = plt.subplots(figsize=(30, 30))

# Positioning of bars on the x-axis for prediction points
x = np.arange(prediction_points)

# Bar width and space between feature groups within each prediction point
bar_width = 0.15
space_between_features = 0.05

# Colors for the different time steps in each segment
colors = plt.cm.viridis(np.linspace(0, 1, time_steps))

# Create a dictionary to store the bars for the legend
legend_bars = {}

# Plotting stacked bars for each feature within each prediction point
for i, feature in enumerate(features):
    for j in range(time_steps):
        # Calculate the position for the bars for each feature
        pos = x + i * (bar_width + space_between_features)
        # Create a bar plot for the current time step
        if j == 0:
            # Create the bottom value for stacked bars
            bottom_values = np.zeros(prediction_points)
        else:
            bottom_values = np.sum(data[:, :j, i], axis=1)

        bars = ax.bar(pos, data[:, j, i], bar_width, bottom=bottom_values, color=colors[j])

        # Add a bar for the first occurrence of each time step to the legend
        if j not in legend_bars:
            legend_bars[j] = bars

# Labeling
ax.set_xlabel('Prediction Points')
ax.set_ylabel('Feature Attribution')
ax.set_title('Stacked Bar Plot of Feature Attributions for Each Prediction Point')
ax.set_xticks(x + (bar_width + space_between_features) * (len(features) / 2))  # Adjust x-axis labels for clarity
ax.set_xticklabels([f'Pred {i+1}' for i in range(prediction_points)])

# Adjusting the legend to show each time step color
handles = [legend_bars[j] for j in range(time_steps)]
labels = [f'Time Step {j+1}' for j in range(time_steps)]
ax.legend(handles=handles, labels=labels, loc='upper right', title='Time Steps')

plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assuming you extracted feature attributions correctly
feat_atts = [res['results'][i][(0, 0)]['attributions'].detach().numpy() for i in range(2923, 2933)]

# Number of prediction points, features, and time steps (adjusted based on your data)
prediction_points = len(feat_atts)
features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
time_steps = feat_atts[0].shape[0]  # Number of time steps from the shape of the feature attributions

# Convert the list of arrays into a 3D NumPy array for easy plotting
data = np.array(feat_atts)  # Shape: (prediction_points, time_steps, features)

# Create a plot
fig, ax = plt.subplots(figsize=(12, 6))

# Positioning of bars on the x-axis for prediction points
x = np.arange(prediction_points)

# Bar width and space between feature groups within each prediction point
bar_width = 0.15
space_between_features = 0.05

# Custom colors for the different time steps
colors = [
    'red', 'green', 'orange', 'grey', 'yellow', 'purple', 'pink', 'blue',
    '#00FF00', '#EED8AE', '#FF5733'
]

# Create a dictionary to store the bars for the legend
legend_bars = {}

# Plotting stacked bars for each feature within each prediction point
for i, feature in enumerate(features):
    for j in range(time_steps):
        # Calculate the position for the bars for each feature
        pos = x + i * (bar_width + space_between_features)
        # Create a bar plot for the current time step
        if j == 0:
            # Create the bottom value for stacked bars
            bottom_values = np.zeros(prediction_points)
        else:
            bottom_values = np.sum(data[:, :j, i], axis=1)

        bars = ax.bar(pos, data[:, j, i], bar_width, bottom=bottom_values, color=colors[j])

        # Add a bar for the first occurrence of each time step to the legend
        if j not in legend_bars:
            legend_bars[j] = bars

# Labeling
ax.set_xlabel('Prediction Points')
ax.set_ylabel('Feature Attribution')
ax.set_title('Stacked Bar Plot of Feature Attributions for Each Prediction Point')
ax.set_xticks(x + (bar_width + space_between_features) * (len(features) / 2))  # Adjust x-axis labels for clarity
ax.set_xticklabels([f'Pred {i+1}' for i in range(prediction_points)])

# Adjusting the legend to show each time step color
handles = [legend_bars[j] for j in range(time_steps)]
labels = [f'Time Step {j+1}' for j in range(time_steps)]
ax.legend(handles=handles, labels=labels, loc='upper right', title='Time Steps')

plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create dummy data
np.random.seed(0)
data = pd.DataFrame({
    'individual': [f"Mister {i+1}" for i in range(60)],
    'group': ['A']*10 + ['B']*30 + ['C']*14 + ['D']*6,
    'value1': np.random.randint(10, 101, 60),
    'value2': np.random.randint(10, 101, 60),
    'value3': np.random.randint(10, 101, 60),
})

# Transform data to long format
data_long = pd.melt(data, id_vars=['individual', 'group'], value_vars=['value1', 'value2', 'value3'], var_name='observation', value_name='value')

# Add empty bars to separate groups
empty_bar = 2
nObsType = len(data_long['observation'].unique())
to_add = pd.DataFrame(np.nan, index=range(empty_bar * len(data_long['group'].unique()) * nObsType), columns=data_long.columns)
to_add['group'] = np.repeat(data_long['group'].unique(), empty_bar * nObsType)
data_long = pd.concat([data_long, to_add]).sort_values(by=['group', 'individual']).reset_index(drop=True)
data_long['id'] = np.repeat(np.arange(1, len(data_long) // nObsType + 1), nObsType)

# Get label positions
label_data = data_long.groupby(['id', 'individual']).agg(tot=('value', 'sum')).reset_index()
number_of_bar = len(label_data)
angle = 90 - 360 * (label_data['id'] - 0.5) / number_of_bar
label_data['hjust'] = np.where(angle < -90, 1, 0)
label_data['angle'] = np.where(angle < -90, angle + 180, angle)

# Prepare base lines
base_data = data_long.groupby('group').agg(start=('id', 'min'), end=('id', 'max')).reset_index()
base_data['title'] = (base_data['start'] + base_data['end']) / 2

# Prepare grid lines
grid_data = base_data.copy()
grid_data['end'] = grid_data['end'].shift(-1).fillna(grid_data['end']) + 1
grid_data['start'] = grid_data['start'] - 1
grid_data = grid_data.iloc[:-1]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
cmap = plt.get_cmap('viridis', len(data_long['observation'].unique()))

# Stacked bars
for i, obs in enumerate(data_long['observation'].unique()):
    subset = data_long[data_long['observation'] == obs]
    bars = ax.bar(np.deg2rad(subset['id'] * 360 / number_of_bar), subset['value'], width=np.deg2rad(360 / number_of_bar), color=cmap(i), alpha=0.5, edgecolor='none', label=obs, bottom=subset.groupby('id')['value'].cumsum().shift(fill_value=0))

# Grid lines
for _, row in grid_data.iterrows():
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [0, 0], color='grey', alpha=0.8, linewidth=0.6)
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [50, 50], color='grey', alpha=0.8, linewidth=0.6)
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [100, 100], color='grey', alpha=0.8, linewidth=0.6)
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [150, 150], color='grey', alpha=0.8, linewidth=0.6)
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [200, 200], color='grey', alpha=0.8, linewidth=0.6)

# Labels
for _, row in label_data.iterrows():
    ax.text(np.deg2rad(row['id'] * 360 / number_of_bar), row['tot'] + 10, row['individual'], ha='center', va='bottom', color='black', fontsize=12, fontweight='bold', rotation=row['angle'])

# Base lines and group labels
for _, row in base_data.iterrows():
    ax.plot(np.deg2rad([row['start'] * 360 / number_of_bar, row['end'] * 360 / number_of_bar]), [-5, -5], color='black', alpha=0.8, linewidth=0.6)
    ax.text(np.deg2rad(row['title'] * 360 / number_of_bar), -18, row['group'], ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

# Final touches
ax.set_ylim(-150, data_long['value'].max() + 50)
ax.set_yticks([])
ax.set_xticks([])
ax.set_rlabel_position(-45)

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

# Save the plot
plt.savefig("output.png", bbox_inches='tight')
plt.show()








import numpy as np
import matplotlib.pyplot as plt

time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
feat = ['x1', 'x2', 'x3', 'x4']
colors = ['red', 'orange', 'blue', 'purple', 'pink', 'green', 'yellow', 'grey', '#00FF00', '#EED8AE', '#FF5733']
values = np.array([
    [-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
    [-1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03],
    [-1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04],
    [-2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05],
    [-2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02],
    [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
    [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
    [-1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04],
    [-1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03],
    [-1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03],
    [-1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03]
])
print(values)
# Store original time indices
sorted_indices = np.argsort(values, axis=0)[::-1]

sorted_values = np.zeros_like(values)
sorted_times = np.zeros_like(values, dtype=int)

for col in range(values.shape[1]):
    sorted_values[:, col] = values[sorted_indices[:, col], col]
    sorted_times[:, col] = np.array(time)[sorted_indices[:, col]]

fig, ax = plt.subplots(figsize=(12, 6))

# Initialize current positions for positive and negative values
positive_starts = np.zeros_like(feat, dtype=float)
negative_starts = np.zeros_like(feat, dtype=float)

legend_handles = []

for i in range(len(time)):
    width = sorted_values[i, :]

    for j in range(len(width)):
        color_idx = time.index(sorted_times[i, j])
        color = colors[color_idx]

        if width[j] > 0:
            left = positive_starts[j]
            positive_starts[j] += width[j]

            # Plot the bar
            bar = ax.barh(feat[j], width[j], left=left, height=0.5, color=color)

        elif width[j] < 0:
            left = negative_starts[j]
            negative_starts[j] += width[j]

            # Plot the bar
            bar = ax.barh(feat[j], width[j], left=left, height=0.5, color=color)

    # Store the handle of the last bar plotted for each time
    legend_handles.append(bar[0])

# Adding a vertical line at x=0
#ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Setting y-axis ticks and labels
ax.set_yticks(range(len(feat)))
ax.set_yticklabels(feat)
ax.set_ylabel('Features')
ax.set_xlabel('Feature attributions')


# Adding the legend
ax.legend(legend_handles, time, title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/leah/Downloads/Deeplearn_data_global.pkl'
path = '/home/leah/projects/KnowIt/leahs-synth-demo/leahs_mlp/interpretations/DeepLiftShap-eval-success-100-True-(2923, 3023).pickle'
# Open the pickle file for reading
with open(path, 'rb') as file:
    # Load the data from the pickle file
    dta = pickle.load(file)
print(dta)

time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
feat = ['x1', 'x2', 'x3', 'x4']
#col = ['red', 'orange', 'green', 'yellow']
colors = ['red', 'orange', 'blue', 'purple', 'pink', 'green', 'yellow', 'grey', '#00FF00', '#EED8AE', '#FF5733']
values = np.array([
        [-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
        [-1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03],
        [-1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04],
        [-2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05],
        [-2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02],
        [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
        [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
        [-1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04],
        [-1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03],
        [-1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03],
        [-1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03]
])
print(values)
# Sort each column in descending order
#sorted_values = np.sort(values, axis=0)[::-1]

fig, ax = plt.subplots(figsize=(12, 6))

# Initialize current positions for positive and negative values
positive_starts = np.zeros_like(feat, dtype=float)
negative_starts = np.zeros_like(feat, dtype=float)

legend_handles = []

for i in range(len(time)):
    width = values[i, :]

    for j in range(len(width)):
        if width[j] > 0:
            left = positive_starts[j]
            positive_starts[j] += width[j]

            # Plot the bar
            bar = ax.barh(feat[j], width[j], left=left, height=0.5, color=colors[i])

        elif width[j] < 0:
            left = negative_starts[j]
            negative_starts[j] += width[j]

            # Plot the bar
            bar = ax.barh(feat[j], width[j], left=left, height=0.5, color=colors[i])

    # Store the handle of the last bar plotted for each time
    legend_handles.append(bar[0])

# Adding a vertical line at x=0
#ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Setting y-axis ticks and labels
ax.set_yticks(range(len(feat)))
ax.set_yticklabels(feat)
ax.set_ylabel('Features')
ax.set_xlabel('Feature attributions')
# Adding the legend
ax.legend(legend_handles, time, title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
feat = ['x1', 'x2', 'x3', 'x4']
#col = ['red', 'orange', 'green', 'yellow']
colors = ['red', 'orange', 'blue', 'purple', 'pink', 'green', 'yellow', 'grey', '#00FF00', '#EED8AE', '#FF5733']
values = np.array([
    [-0.02, -0.046, -0.036, -0.012],
    [-0.02, -0.037, -0.103, 0.004],
    [-0.018, -0.038, -0.02, 0.0003],
    [-0.021, -0.327, -0.018, 0.00002],
    [-0.028, -0.028, -0.009, 0.012],
    [-0.132, -0.016, -0.004, -0.005],
    [-0.024, -0.016, -0.006, -0.004],
    [-0.019, -0.013, 0.001, 0.0007],
    [-0.01, -0.008, 0.001, 0.006],
    [-0.014, -0.004, 0.002, -0.002],
    [-0.011, 0.007, 0.003, 0.003]
])

# Sort each column in descending order
sorted_values = np.sort(values, axis=0)[::-1]

fig, ax = plt.subplots(figsize=(12, 6))

# Initialize current positions for positive and negative values
positive_starts = np.zeros_like(feat, dtype=float)
negative_starts = np.zeros_like(feat, dtype=float)

legend_handles = []

for i in range(len(time)):
    width = sorted_values[i, :]

    for j in range(len(width)):
        if width[j] > 0:
            left = positive_starts[j]
            positive_starts[j] += width[j]

            # Plot the bar
            ax.barh(feat[j], width[j], left=left, height=0.5, color=colors[i], edgecolor='black')

        elif width[j] < 0:
            left = negative_starts[j]
            negative_starts[j] += width[j]

            # Plot the bar
            ax.barh(feat[j], width[j], left=left, height=0.5, color=colors[i], edgecolor='black')

# Adding a vertical line at x=0
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Setting y-axis ticks and labels
ax.set_yticks(range(len(feat)))
ax.set_yticklabels(feat)
ax.set_xlabel('Values')
ax.set_title('Sorted Horizontal Bars with Separation Lines')
ax.legend(legend_handles, time, title='Time', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# feat = ['x1', 'x2', 'x3', 'x4']
# #col = ['red', 'orange', 'green', 'yellow']
# color = ['red', 'orange', 'blue', 'purple', 'pink', 'green', 'yellow', 'grey', '#00FF00', '#EED8AE', '#FF5733']
# values = np.array([
#     [-0.02, -0.046, -0.036, -0.012],
#     [-0.02, -0.037, -0.103, 0.004],
#     [-0.018, -0.038, -0.02, 0.0003],
#     [-0.021, -0.327, -0.018, 0.00002],
#     [-0.028, -0.028, -0.009, 0.012],
#     [-0.132, -0.016, -0.004, -0.005],
#     [-0.024, -0.016, -0.006, -0.004],
#     [-0.019, -0.013, 0.001, 0.0007],
#     [-0.01, -0.008, 0.001, 0.006],
#     [-0.014, -0.004, 0.002, -0.002],
#     [-0.011, 0.007, 0.003, 0.003]
# ])
#
# # Sort each row in descending order
# sorted_values = np.sort(values, axis=0)[::-1]
# print(sorted_values)
#
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # Initialize current positions for positive and negative values
# positive_starts = np.zeros_like(feat, dtype=float)
# negative_starts = np.zeros_like(feat, dtype=float)
#
# for i,(time,color) in enumerate(zip(time,color)):
#     width = sorted_values[i, :]
#
#     for j in range(len(width)):
#         if width[j] > 0:
#             left = positive_starts[j]
#             positive_starts[j] += width[j]
#
#             # Plot the bar
#             ax.barh(feat[j], width[j], left=left, height=0.5, color=color[i], edgecolor='black')
#
#         elif width[j] < 0:
#             left = negative_starts[j]
#             negative_starts[j] += width[j]
#
#             # Plot the bar
#             ax.barh(feat[j], width[j], left=left, height=0.5, color=color[i], edgecolor='black')
#
# # Adding a vertical line at x=0
# ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
#
# # Setting y-axis ticks and labels
# ax.set_yticks(range(len(feat)))
# ax.set_yticklabels(feat)
# ax.set_xlabel('Values')
# ax.set_title('Sorted Horizontal Bars with Separation Lines')
#
# plt.show()




time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
feat = ['x1', 'x2', 'x3', 'x4']
col = ['red', 'orange', 'green', 'yellow']
values = np.array([
    [-0.02, -0.046, -0.036, -0.012],
    [-0.02, -0.037, -0.103, 0.004],
    [-0.018, -0.038, -0.02, 0.0003],
    [-0.021, -0.327, -0.018, 0.00002],
    [-0.028, -0.028, -0.009, 0.012],
    [-0.132, -0.016, -0.004, -0.005],
    [-0.024, -0.016, -0.006, -0.004],
    [-0.019, -0.013, 0.001, 0.0007],
    [-0.01, -0.008, 0.001, 0.006],
    [-0.014, -0.004, 0.002, -0.002],
    [-0.011, 0.007, 0.003, 0.003]
])

fig, ax = plt.subplots(figsize=(12, 6))

# Initialize current positions for positive and negative values
positive_starts = np.zeros_like(time, dtype=float)
negative_starts = np.zeros_like(time, dtype=float)

for i in range(len(time)):
    width = values[i, :]

    for j in range(len(width)):
        if width[j] > 0:
            left = positive_starts[j]
            positive_starts[j] += width[j]

            # Plot the bar
            ax.barh(feat[j], width[j], left=left, height=0.5, color=col[j])

        elif width[j] < 0:
            # If the width is negative

            left = negative_starts[j]
            negative_starts[j] += width[j]

            # Plot the bar
            ax.barh(feat[j], width[j], left=left, height=0.5, color=col[j])
plt.show()
# Show only one legend entry per category
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# ax.legend(unique_labels.values(), unique_labels.keys())


# values= np.array([
#      [2, 4, 6, 8],
#      [10, 12, 14, 16],
#      [18, 20, 22, 24]
#  ])
# feat = ['x1', 'x2', 'x3', 'x4']
# col = ['red', 'orange', 'green', 'yellow']
#
#
# fig, ax = plt.subplots(figsize=(10, 5))
#
# for i in range(values.shape[1]):
#     dat = values[i, :]
#     sort_indices = np.argsort(-dat)
#     dat_sort = dat[sort_indices]
#     cum_sort = values[:, i].cumsum()[sort_indices]
#
#     start = cum_sort - dat_sort
#
#
#     ax.barh(feat[i], dat_sort, left= start, height=0.5, color=col[i], label=feat[i])
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

values = np.array([
    [2, 4, 6, 8],
    [10, 12, 14, 16],
    [18, 20, 22, 24]
])
feat = ['x1', 'x2', 'x3', 'x4']
col = ['red', 'orange', 'green', 'yellow']

fig, ax = plt.subplots(figsize=(10, 5))

# Loop over each feature
for i in range(len(feat)):
    dat = values[:, i]

    # Sort the data and corresponding cumulative values
    sorted_indices = np.argsort(dat)
    dat_sort = dat[sorted_indices]
    cum_sort = values[:, i].cumsum()[sorted_indices]

    # Calculate the start positions
    start = cum_sort - dat_sort

    # Plot each feature's bar horizontally
    ax.barh(y=feat[i], width=dat_sort, left=start, height=0.5, color=col[i], label=feat[i])

# Set y-ticks to be the feature names
ax.set_yticks(range(len(feat)))
ax.set_yticklabels(feat)

# Add legend
ax.legend()

plt.show()

import sys
import torch
import shap
import tensorflow as tf
time = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
col = ['red','orange', 'green', 'yellow']
feat = ['x1', 'x2', 'x3', 'x4']
values= np.array([
    [-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
     [-1.9962e-02, -3.6710e-02, -1.0312e-01, 4.7214e-03],
     [-1.8299e-02, -3.8295e-02, -2.0489e-02, 2.8094e-04],
     [-2.1214e-02, -3.2716e-01, -1.8155e-02, 2.3168e-05],
     [-2.7828e-02, -2.8243e-02, -9.0340e-03, 1.2185e-02],
     [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
     [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
     [-1.8900e-02, -1.3745e-02, 1.6986e-03, 7.4371e-04],
     [-1.0523e-02, -8.6838e-03, 1.1933e-03, 6.3727e-03],
     [-1.4640e-02, -4.9247e-03, 2.0039e-03, -1.8333e-03],
     [-1.1908e-02, 7.0504e-03, 3.2147e-03, 3.9074e-03]
])
print(values)
dat_cum = values.cumsum(axis=1)
fig, ax = plt.subplots(figsize=(28,3))

for i, (category, color) in enumerate(zip(feat, col)):
    width= values[:,i]
    starts= dat_cum[:,i] - width
    ax.barh(time, width, left=starts, height=0.5,
            color=color, label= category)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

year = 2017  # Single year for y-coordinate
dat = np.array([10, 24, 37, 4, 15, 39, 22, 13])
feat = ['x1', 'x2', 'x3', 'x4']
col = ['red', 'orange', 'yellow', 'green']

# Repeat feat and col to match data length
list2 = np.tile(feat, 2)
list3 = np.tile(col, 2)

dat_cum = dat.cumsum(axis=0)
starts = dat_cum - dat

fig, ax = plt.subplots(figsize=(9.2, 2))  # Adjust figure height for smaller bars

# Plot each bar segment individually with smaller height
bar_height = 0.2  # Smaller height for bars
for i in range(len(dat)):
    ax.barh(year, dat[i], left=starts[i], height=bar_height, color=list3[i])
plt.show()
import numpy as np
import matplotlib.pyplot as plt

year = 2017
dat = np.array([10, 24, 37, 4, 15, 39, 22, 13])
feat = ['x1', 'x2', 'x3', 'x4']
col = ['red', 'orange', 'yellow', 'green']

# Repeat feat and col to match data length
list2 = np.tile(feat, 2)
list3 = np.tile(col, 2)

dat_cum = dat.cumsum(axis=0)
starts = dat_cum - dat

fig, ax = plt.subplots(figsize=(9.2, 2))  # Adjust figure height for smaller bars

# Plot each bar segment individually with smaller height
bar_height = 0.2  # Smaller height for bars
for i in range(len(dat)):
    ax.barh(year, dat[i], left=starts[i], height=bar_height, color=list3[i])

# Annotate each bar segment with the corresponding feature
for i in range(len(dat)):
    width = dat[i]
    ax.text(starts[i] + width / 2, year, list2[i], ha='center', va='center', color='white')

# Ensure only one y-tick is shown
ax.set_yticks([])
#ax.set_yticklabels([year])
ax.set_ylim(year - 0.5, year + 0.5)  # Adjust the y-axis limits to make bars centered
ax.set_xlabel('Feature attributions')


# Show only one legend entry per unique feature
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()





year= 2017
dat= np.array([10,24,37,4,15,39,22,13])
feat= ['x1','x2','x3','x4']
list2 = np.tile(feat, 2)
col=['red', 'orange', 'yellow', 'green']
list3= np.tile(col,2)
dat_cum= dat.cumsum(axis=0)
starts = dat_cum-dat
fig,ax=plt.subplots(figsize=(9.2,5))

for i in range(len(dat)):
    ax.barh(year,dat,left=starts,height=0.05,color=list3,label=list2)
plt.show()



year= np.array([2003,2009,2015,2021,2027])
wb= np.array([24.7,16.2,15.2,17.9,21.7])
veg= np.array([43.2,49.5,50.5,44.5,43.0])
sett= np.array([4.3,5.6,6.0,13.6,13.7])

p1 = plt.barh(year, wb, height=2.8, label='Waterbodies', color='r')
p2 = plt.barh(year, veg, height=2.8, left= wb, color='#00FF00', label='Vegetarians')
p3 = plt.barh(year, sett , height=2.8, left= veg, color='b',label ='Settlement')

plt.xlabel('Percentage')
plt.ylabel('Year')
plt.title('Horizontal Stacked Bar Plot')

# plt.bar_label(p1, label_type='center',fontsize=8)
# plt.bar_label(p2, label_type='center',fontsize=8)
# plt.bar_label(p3, label_type='center',fontsize=8)

plt.yticks(year)
plt.xlim(0,100)
plt.legend(loc=(0.04,-0.3),ncol=2)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Example data
y = ['A', 'B', 'C']
width = [10, 20, 15]
height = 0.8
left = [0, 0, 0]
color = ['red', 'green', 'blue']
label = ['Label A', 'Label B', 'Label C']

# Create horizontal bar chart
fig, ax = plt.subplots()
ax.barh(y=y, width=width, height=height, left=left, color=color, label=label)

# Adding labels
for i, (width_value, color_value) in enumerate(zip(width, color)):
    ax.text(width_value + 0.5, i, f'{width_value:.2f}', va='center', ha='left', color=color_value)

ax.set_xlabel('Values')
ax.set_title('Horizontal Bar Chart Example')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Define the feature names and their attributions for a single instance
feature_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
attributions = [10, 15, 17, 32, 26]  # Attributions for the single instance


def plot_single_instance_attributions(attributions, feature_names):
    """
    Parameters
    ----------
    attributions : list
        A list of feature attributions for a single instance.
        It is assumed the list contains the same number of entries as *feature_names*.
    feature_names : list of str
        The feature labels.
    """
    labels = ['Instance 1']  # Label for the single instance
    data = np.array(attributions).reshape(1, -1)  # Reshape to 2D array
    data_cum = data.cumsum(axis=1)
    feature_colors = plt.cm.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 1.5))  # Adjusted height for a single instance
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (featurename, color) in enumerate(zip(feature_names, feature_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=featurename, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

        # Manually add labels
        for rect in rects:
            width = rect.get_width()
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                    f'{width}', ha='center', va='center', color=text_color)

    ax.legend(ncol=len(feature_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


plot_single_instance_attributions(attributions, feature_names)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate some data for 1 row and 5 columns
data = np.random.rand(1, 45)
print(data)

# Create a figure and axis
fig, ax = plt.subplots()

# Display the data as an image
cax = ax.imshow(data, cmap='inferno', aspect='auto')

# Add annotations
for j in range(data.shape[1]):
    value = data[0, j]
    color = (   
                'white' if value < 0.1 else
                'black' if value < 0.2 else
                'red' if value < 0.3 else
                'green' if value < 0.4 else
                'purple' if value <= 0.5 else
                'yellow')
    # First text annotation (main value)
    ax.text(j, 0, f'{value:.2f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    # Second text annotation (position), slightly offset
    ax.text(j, 0.3, f'({0}, {j})', ha='center', va='center', color=color, fontsize=8)

# Add a colorbar
plt.colorbar(cax)

# Set title and labels
ax.set_title('Heatmap with Annotations (1 Row)')
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')

# Adjust y-axis to show only the single row
ax.set_yticks([0])
ax.set_yticklabels(['0'])

# Show the plot
plt.show()




from math import pi

import pandas as pd

from bokeh.models import BasicTicker, PrintfTickFormatter
from bokeh.plotting import figure, show
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import linear_cmap
#
data['Year'] = data['Year'].astype(str)
data = data.set_index('Year')
data.drop('Annual', axis=1, inplace=True)
data.columns.name = 'Month'
#
years = list(data.index)
months = list(reversed(data.columns))
#
# reshape to 1D array or rates with a month and year for each row.
df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
#
# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
#
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
#
p = figure(title=f"US Unemployment ({years[0]} - {years[-1]})",
            x_range=years, y_range=months,
            x_axis_location="above", width=900, height=400,
            tools=TOOLS, toolbar_location='below',
           tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])
#
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "7px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3
#
r = p.rect(x="Year", y="Month", width=1, height=1, source=df,
            fill_color=linear_cmap("rate", colors, low=df.rate.min(), high=df.rate.max()),
            line_color=None)
#
p.add_layout(r.construct_color_bar(
    major_label_text_font_size="7px",
    ticker=BasicTicker(desired_num_ticks=len(colors)),
     formatter=PrintfTickFormatter(format="%d%%"),
     label_standoff=6,
    border_line_color=None,
     padding=5,
), 'right')

show(p)



# import pandas as pd
# from bokeh.layouts import column
# from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource
#
# # Your data dictionary
# data = {
#     'Feature1': [0.2, 0.3, 0.5, 0.7],
#     'Feature2': [0.1, 0.4, 0.6, 0.8],
#     'Feature3': [0.3, 0.5, 0.7, 0.9],
#     'Feature4': [0.4, 0.6, 0.8, 1.0]
# }
#
# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)
#
# # Create x-values as index for the plots
# x_values = df.index
#
# # Create a list to hold the individual plots
# plots = []
#
# # Create individual plots for each feature
# for feature in df.columns:
#     p = figure(height=300, width=800, title=feature)
#     p.line(x_values, df[feature], line_width=2, legend_label=feature)
#     p.legend.location = "top_left"
#     p.yaxis.axis_label = 'Value'
#     plots.append(p)
#
# # Arrange the plots in a column layout
# layout = column(*plots)
#
# # Show the layout
# show(layout)







# path= '/home/leah/projects/KnowIt/leahs-synth-demo/leahs_mlp/bestmodel-epoch=7-valid_loss=0.00.ckpt'
# # Load the model from the checkpoint file
# #sys.path.append('/home/leah/projects/KnowIt/leahs-synth-demo/leahs_mlp/model_args.yaml')
import matplotlib.pyplot as plt

# Sample data
data = [0.1, 0.3, 0.5, 0.7, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# Create a new figure and axis
fig, ax = plt.subplots()

# Draw rugplot
for point in data:
    ax.axvline(point, ymin=0, ymax=0.05, color='blue')  # Adjust ymin and ymax for line length

# Set labels and title
ax.set_xlabel('Data')
ax.set_ylabel('Frequency')
ax.set_title('Rugplot')

# Hide y-axis
ax.yaxis.set_visible(False)

# Show plot
plt.show()

# model = torch.load(path)

#print(model)
iris = datasets.load_iris()
column=iris.target_names
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()

def plot_single_pair(X, y, features, colormap, ax):
    for i in range(0, 4):
        for j in range(0, 4):
            feature_count = len(features)
            if i == j:
                tdf = pd.DataFrame(X[:, i], columns=[features[i]])
                tdf['target'] = y
                for c in colormap.keys():
                    tdf_filtered = tdf.loc[tdf['target'] == c]
                    ax[i, j].hist(tdf_filtered[features[i]], color=colormap[c], bins=30)
            else:
                tdf = pd.DataFrame(X[:, [i, j]], columns=[features[i], features[j]])
                tdf['target'] = y
                for c in colormap.keys():
                    tdf_filtered = tdf.loc[tdf['target'] == c]
                    ax[i, j].scatter(x=tdf_filtered[features[j]], y=tdf_filtered[features[i]], color=colormap[c])

# Create the figure and axes
feature_count = len(iris.feature_names)
fig, ax = plt.subplots(nrows=feature_count, ncols=feature_count, figsize=(16, 16))

# Plot pairwise
plot_single_pair(iris.data, iris.target, iris.feature_names, colormap={0: "red", 1: "green", 2: "blue"}, ax=ax)

plt.show()

x= [[-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
        [-1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03],
        [-1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04],
        [-2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05],
        [-2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02],
        [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
        [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
        [-1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04],
        [-1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03],
        [-1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03],
        [-1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03]]
arr3 = np.array(x)
tot2= pd.DataFrame(arr3, columns=['x1', 'x2', 'x3', 'x4'])


print(tot2)
column_sums = tot2.sum()

print(column_sums)


# from bokeh.palettes import Spectral4
# from bokeh.plotting import figure, show
#
#
#
# # Convert the dictionary to a DataFrame
#
#
# # Assuming the x-values are the indices of the DataFrame
# x_values = tot2.index
#
# # Create a figure
# p = figure(width=800, height=250)
# p.title.text = 'Click on legend entries to hide the corresponding lines'
#
# # Plot each feature
# for name, color in zip(tot2.columns, Spectral4):
#     p.line(x_values, tot2[name], line_width=2, color=color, alpha=0.8, legend_label=name)
#
# # Configure the legend
# p.legend.location = "bottom_right"
# p.legend.click_policy = "hide"
#
# # Show the plot
# show(p)

# import numpy as np
# import pandas as pd
# from bokeh.layouts import column
# from bokeh.models import ColumnDataSource, RangeTool
# from bokeh.plotting import figure, show
#
# # Your data dictionary
#
# source = ColumnDataSource(data=tot2)
#
# # Create x-values as index for the range tool
# x_values = np.arange(len(tot2))
#
# p = figure(height=300, width=800, tools="xpan", toolbar_location=None,
#            x_axis_type="linear", x_axis_location="above",
#            background_fill_color="#efefef", x_range=(x_values[1], x_values[3]))
#
# # Plot lines for each feature
# for feature in tot2.columns:
#     p.line(x_values, tot2[feature], line_width=2, legend_label=feature, color=Spectral4[tot2.columns.get_loc(feature)])
#
# p.yaxis.axis_label = 'Value'
# p.legend.location = "bottom_right"
# p.legend.click_policy="hide"
#
#
# select = figure(title="Drag the middle and edges of the selection box to change the range above",
#                 height=200, width=800, y_range=p.y_range,
#                 x_axis_type="linear", y_axis_type=None,
#                 tools="", toolbar_location=None, background_fill_color="#efefef")
#
# range_tool = RangeTool(x_range=p.x_range)
# range_tool.overlay.fill_color = "navy"
# range_tool.overlay.fill_alpha = 0.2
#
# for feature in tot2.columns:
#     select.line(x_values, tot2[feature], legend_label=feature, color=Spectral4[tot2.columns.get_loc(feature)])
#
# select.yaxis.axis_label = 'Value'
# select.legend.location = "bottom_right"
# select.legend.click_policy="hide"
#
# select.ygrid.grid_line_color = None
# select.add_tools(range_tool)
#
# show(column(p, select))



#def practice(X, ax):
fig, ax=plt.subplots(4,4,figsize=(16,16))
features = ['x1', 'x2', 'x3', 'x4']
for i in range(0,4):
    for j in range(0,4):

        if i==j:
            ex=pd.DataFrame(tot2.iloc[:, i], columns=[features[i]])
            ax[i,j].plot(ex)
        else:
            ex=pd.DataFrame(tot2.iloc[:,[i,j]])
            ax[i,j].scatter(x=ex[features[j]],y=ex[features[i]])

        if i == len(features) - 1:
            ax[i, j].set(xlabel=features[j], ylabel='')
        if j == 0:
            if i == len(features) - 1:
                ax[i, j].set(xlabel=features[j], ylabel=features[i])
            else:
                ax[i, j].set(xlabel='', ylabel=features[i])

plt.show()


#fig, ax=plt.subplots(4,4,figsize=(16,16))
#practice(tot,ax=ax)
#plt.show()

path1 ='data.pickle'
with open(path1, 'rb') as f:
    features = pickle.load(f)

print(features)
features_1= features[0]
print(features_1)

path2='please.pickle'
with open(path2, 'rb') as f:
    baselines = pickle.load(f)

#print(baselines)
baseline_1=baselines[0]
#print(baseline_1)

x=np.array([-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02,
        -1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03,
        -1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04,
        -2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05,
        -2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02,
        -1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03,
        -2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03,
        -1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04,
        -1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03,
        -1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03,
        -1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03])

reshaped_data = x.reshape(1, 44)
x_sort=np.sort(reshaped_data)
print(x_sort)
fig, ax = plt.subplots(figsize=(20, 50))
ax.imshow(x_sort, cmap="inferno", aspect=2)
ax.text(0, 0, f'{4}', ha='center', va='center', color='green', fontsize=10, fontweight='bold')
ax.text(0, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)


ax.text(1, 0, f'{10}', ha='center', va='center', color='magenta', fontsize=10, fontweight='bold')
ax.text(1, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(2, 0, f'{8}', ha='center', va='center', color='pink', fontsize=10, fontweight='bold')
ax.text(2, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(3, 0, f'{1}', ha='center', va='center', color='blue', fontsize=10, fontweight='bold')
ax.text(3, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(4, 0, f'{10}', ha='center', va='center', color='magenta', fontsize=10, fontweight='bold')
ax.text(4, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(5, 0, f'{10}', ha='center', va='center', color='magenta', fontsize=10, fontweight='bold')
ax.text(5, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(6, 0, f'{9}', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')
ax.text(6, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(7, 0, f'{8}', ha='center', va='center', color='pink', fontsize=10, fontweight='bold')
ax.text(7, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(8, 0, f'{7}', ha='center', va='center', color='purple', fontsize=10, fontweight='bold')
ax.text(8, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(9, 0, f'{7}', ha='center', va='center', color='purple', fontsize=10, fontweight='bold')
ax.text(9, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(10, 0, f'{2}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
ax.text(10, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(11, 0, f'{3}', ha='center', va='center', color='orange', fontsize=10, fontweight='bold')
ax.text(11, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(12, 0, f'{9}', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')
ax.text(12, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(13, 0, f'{6}', ha='center', va='center', color='cyan', fontsize=10, fontweight='bold')
ax.text(13, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(14, 0, f'{5}', ha='center', va='center', color='brown', fontsize=10, fontweight='bold')
ax.text(14, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(15, 0, f'{9}', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')
ax.text(15, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(16, 0, f'{5}', ha='center', va='center', color='brown', fontsize=10, fontweight='bold')
ax.text(16, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(17, 0, f'{6}', ha='center', va='center', color='cyan', fontsize=10, fontweight='bold')

ax.text(18, 0, f'{8}', ha='center', va='center', color='pink', fontsize=10, fontweight='bold')
ax.text(18, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(19, 0, f'{4}', ha='center', va='center', color='green', fontsize=10, fontweight='bold')
ax.text(19, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(20, 0, f'{8}', ha='center', va='center', color='pink', fontsize=10, fontweight='bold')
ax.text(20, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(21, 0, f'{10}', ha='center', va='center', color='magenta', fontsize=10, fontweight='bold')
ax.text(21, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(22, 0, f'{0}', ha='center', va='center', color='red', fontsize=10, fontweight='bold')
ax.text(22, 0.3, 'x4', ha='center', va='center', color='white', fontsize=10)

ax.text(23, 0, f'{7}', ha='center', va='center', color='purple', fontsize=10, fontweight='bold')
ax.text(23, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(24, 0, f'{9}', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')
ax.text(24, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(25, 0, f'{5}', ha='center', va='center', color='brown', fontsize=10, fontweight='bold')
ax.text(25, 0.3, 'x2', ha='center', va='center', color='blue',fontsize=10)

ax.text(26, 0, f'{6}', ha='center', va='center', color='cyan', fontsize=10, fontweight='bold')
ax.text(26, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(27, 0, f'{3}', ha='center', va='center', color='orange', fontsize=10, fontweight='bold')
ax.text(27, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(28, 0, f'{2}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
ax.text(28, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(29, 0, f'{7}', ha='center', va='center', color='purple', fontsize=10, fontweight='bold')
ax.text(29, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(30, 0, f'{0}', ha='center', va='center', color='red', fontsize=10, fontweight='bold')
ax.text(30, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)


ax.text(31, 0, f'{1}', ha='center', va='center', color='blue', fontsize=10, fontweight='bold')
ax.text(31, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(32, 0, f'{2}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
ax.text(32, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(33, 0, f'{3}', ha='center', va='center', color='orange', fontsize=10, fontweight='bold')
ax.text(33, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(34, 0, f'{6:}', ha='center', va='center', color='cyan', fontsize=10, fontweight='bold')
ax.text(34, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(35, 0, f'{4}', ha='center', va='center', color='green', fontsize=10, fontweight='bold')
ax.text(35, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(36, 0, f'{4}', ha='center', va='center', color='green', fontsize=10, fontweight='bold')
ax.text(36, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(37, 0, f'{0}', ha='center', va='center', color='red', fontsize=10, fontweight='bold')
ax.text(37, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(38, 0, f'{1}', ha='center', va='center', color='blue', fontsize=10, fontweight='bold')
ax.text(38, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(39, 0, f'{2}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
ax.text(39, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)

ax.text(40, 0, f'{0}', ha='center', va='center', color='red', fontsize=10, fontweight='bold')
ax.text(40, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)


ax.text(41, 0, f'{1}', ha='center', va='center', color='blue', fontsize=10, fontweight='bold')
ax.text(41, 0.3, 'x3', ha='center', va='center', color='black', fontsize=10)

ax.text(42, 0, f'{5}', ha='center', va='center', color='brown', fontsize=10, fontweight='bold')
ax.text(42, 0.3, 'x1', ha='center', va='center', color='red', fontsize=10)

ax.text(43, 0, f'{3}', ha='center', va='center', color='orange', fontsize=10, fontweight='bold')
ax.text(43, 0.3, 'x2', ha='center', va='center', color='blue', fontsize=10)


ax.set_yticks([])
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Data
att = np.array([-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02,
                -1.9962e-02, -3.6710e-02, -1.0312e-01, 4.7214e-03,
                -1.8299e-02, -3.8295e-02, -2.0489e-02, 2.8094e-04,
                -2.1214e-02, -3.2716e-01, -1.8155e-02, 2.3168e-05,
                -2.7828e-02, -2.8243e-02, -9.0340e-03, 1.2185e-02,
                -1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03,
                -2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03,
                -1.8900e-02, -1.3745e-02, 1.6986e-03, 7.4371e-04,
                -1.0523e-02, -8.6838e-03, 1.1933e-03, 6.3727e-03,
                -1.4640e-02, -4.9247e-03, 2.0039e-03, -1.8333e-03,
                -1.1908e-02, 7.0504e-03, 3.2147e-03, 3.9074e-03])

sequence = ['x1', 'x2', 'x3', 'x4']
list1 = np.tile(sequence, 11)

seq = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
rep = 4
list2 = np.repeat(seq, rep)



# Sort the zipped lists
zipped_lists = list(zip(att, list1, list2))
sort_zip = sorted(zipped_lists)



# Unzip sorted lists
a_sorted, b_sorted, c_sorted = zip(*sort_zip)

# Convert to numpy arrays and reshape
attributions = np.array(a_sorted).reshape(1, 44)
features = np.array(b_sorted).reshape(1, 44)
time = np.array(c_sorted).reshape(1, 44)

# Plot
fig, ax = plt.subplots(figsize=(20, 5))  # Adjusted figsize for better view
caw = ax.imshow(attributions, cmap="viridis", aspect=2)

# Add text annotations
for i in range(attributions.shape[1]):
    value1 = features[0, i]
    value2 = time[0, i]
    color1 = (
        'white' if value1 == 'x4' else
        'black' if value1 == 'x3' else
        'blue' if value1 == 'x2' else
        'red'
    )
    color2 = (
        'red' if value2 == -5 else
        'blue' if value2 == -4 else
        'black' if value2 == -3 else
        'white' if value2 == -2 else
        'green' if value2 == -1 else
        'brown' if value2 == 0 else
        'cyan' if value2 == 1 else
        'purple' if value2 == 2 else
        'pink' if value2 == 3 else
        'gray' if value2 == 4 else
        'magenta'
    )
    ax.text(i, 0, f'{features[0, i]}', ha='center', va='center', color=color1, fontsize=10, fontweight='bold')
    ax.text(i, 0.3, time[0, i], ha='center', va='center', color=color2, fontsize=10)

ax.set_yticks([])
plt.colorbar(caw)
plt.show()

att=np.array([-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02,
        -1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03,
        -1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04,
        -2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05,
        -2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02,
        -1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03,
        -2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03,
        -1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04,
        -1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03,
        -1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03,
        -1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03])

sequence = ['x1', 'x2', 'x3', 'x4']
list1 = np.tile(sequence, 11)
print(list1 )

seq = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
rep = 4
list2 = np.repeat(seq, rep)
print(list2)

zipped_lists = list(zip(att, list1, list2))
sort_zip = sorted(zipped_lists)
print(sort_zip)
a_sorted, b_sorted, c_sorted = zip(*sort_zip )
a_sorted = list(a_sorted)
val1=a_sorted[1]
print(val1)
b_sorted = list(b_sorted)
val2=b_sorted[1]
print(val2)
c_sorted = list(c_sorted)
val3=c_sorted[1]
print(val3)

print(a_sorted)
print(b_sorted)
print(c_sorted)
attributions = np.array(a_sorted)
att1= attributions.reshape(1, 44)
features = np.array(b_sorted)
feat = features.reshape(1, 44)
time = np.array(c_sorted)
t = time.reshape(1, 44)

fig, ax0 = plt.subplots(figsize=(20, 50))
caw = ax.imshow(att1, cmap="inferno", aspect=2)
for i in range(45):
    value1 = feat[i]
    value2 = t[i]
    color1 = (
        'white' if value1 == 'x4' else
        'black' if value1 == 'x3' else
        'blue' if value1 == 'x2' else
        'red'
    )
    color2 = (
        'red' if value2 == -5 else
        'blue' if value2 == -4 else
        'black' if value2 == -3 else
        'white' if value2 == -2 else
        'green' if value2 == -1 else
        'brown' if value2 == 0 else
        'cyan' if value2 == 1 else
        'purple' if value2 == 2 else
        'pink' if value2 == 3 else
        'gray' if value2 == 4 else
        'magenta'
    )

    ax0.text(i, 0, f'{feat[i]}', ha='center', va='center', color= color1, fontsize=10, fontweight='bold')
    ax0.text(i, 0.3, t[i], ha='center', va='center', color= color2, fontsize=10)

plt.colorbar(caw)
plt.show()

x= [[-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
        [-1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03],
        [-1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04],
        [-2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05],
        [-2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02],
        [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
        [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
        [-1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04],
        [-1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03],
        [-1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03],
        [-1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03]]






########################################################################################################################
total= [-0.5838,  0.1065,  1.0305,  0.7471,
        -0.5933,  0.1867,  1.1367, -0.5720,
        -0.6103,  0.3042,  1.2145, -0.8367,
        -0.7468,  0.3673,  1.2441, -0.8036,
        -0.8530,  0.4174,  1.2985, -0.7080,
        -0.7835,  0.6232,  1.3483,  0.7774,
        -0.8886,  0.6146,  1.4320,  0.0315,
        -0.8257,  0.7411,  1.3433,  1.0996,
        -0.9944,  0.7576,  1.3255, -0.5640,
        -0.9438,  0.8406,  1.2425, -0.1583,
        -0.8977,  0.8469,  1.2658,  0.5013]
arr4 = np.array(total)
print(x)
dec = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4'])
print(dec)
arr1 = features_1.detach().numpy()
#print(arr1)
feature_dat = pd.DataFrame(arr1, columns=['x1','x2','x3','x4'])
print(feature_dat)

statistics = {
    'Mean': feature_dat.mean(),
    'Median': feature_dat.median(),
    'Standard Deviation': feature_dat.std(),
    'Variance': feature_dat.var(),
    'Skewness': feature_dat.skew(),
    'Kurtosis': feature_dat.kurt(),
    'Maximum': feature_dat.max(),
    'Minimum': feature_dat.min()
}

# Creating the summary table
summary_table = pd.DataFrame(statistics)
print(summary_table)

print(feature_dat)
new_index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
feature_dat.index=new_index

fig, ax55 = plt.subplots()

# Plotting the data
ax55.plot(new_index, feature_dat['x1'], label='x1', color='blue')
ax55.plot(new_index, feature_dat['x2'], label='x2', color='green')
ax55.plot(new_index, feature_dat['x3'], label='x3', color='red')
ax55.plot(new_index, feature_dat['x4'], label='x4', color='orange')

# Setting the x-axis label with customizations
ax55.set_xlabel('Time', fontsize=14, labelpad=15)

# Setting the y-axis label with customizations
ax55.set_ylabel('Features', fontsize=14, labelpad=15)

# Setting the title of the plot
ax55.set_title('A plot displaying the feature values for the various features. ', fontsize=12)

# Display the legend
ax55.legend()

# Show the plot
plt.show()

data = {'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Temperature': [30, 31, 29, 28, 32, 34, 33, 31, 29, 30]}
df = pd.DataFrame(data)

# Creating lagged features
df['Temperature_Lag1'] = df['Temperature'].shift(1)
df['Temperature_Lag2'] = df['Temperature'].shift(2)

# Plot the original and lagged values
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Temperature'], marker='o', label='Temperature')
plt.plot(df['Date'], df['Temperature_Lag1'], marker='x', label='Temperature Lag 1')
plt.plot(df['Date'], df['Temperature_Lag2'], marker='s', label='Temperature Lag 2')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature and Lagged Values Over Time')
plt.legend()
plt.show()


arr1_1=np.round(arr1,2)

tot= pd.DataFrame(arr1, columns=['x1', 'x2', 'x3', 'x4'])
print(tot)
arr2 =  baseline_1.detach().numpy()
tot1= pd.DataFrame(arr2, columns=['x1', 'x2', 'x3', 'x4'])
print(tot1)

arr2_1=np.round(arr2,2)

arr3 = np.array(x)
arr3_1=np.round(arr3,2)
print(arr1_1[0])


shap.plots.force(arr4[0], shap_values= arr3_1[0], features=arr1_1[0], show=True, matplotlib=True, feature_names= ['x1','x2','x3','x4'])






shap.summary_plot(arr3, features=tot)


# Assuming you have your own feature values, baselines, and attributions
# Replace these placeholders with your actual data
  # Your attributions

# Create SHAP values using your data
shap_values1 = shap.Explanation(values=arr3, base_values=arr4, data=tot)
print(shap_values1)
shap.plots.waterfall(shap_values1[0])
#shap dependence plot
shap.dependence_plot(ind=0, shap_values=arr3, features=arr1, feature_names= ['x1', 'x2', 'x3', 'x4'], interaction_index= None)
#shap dependence interaction plot
shap.dependence_plot(ind=0, shap_values=arr3, features=arr1, feature_names= ['x1', 'x2', 'x3', 'x4'], interaction_index=1)
#shap.plots.force(shap_values1[0])
#shap.force_plot(arr4[0], shap_values= arr3[0], features=tot[0],figsize=(3,3))



# Now you can use shap_values for plotting or analysis

# path1= 'synth_1.pickle'
# with open(path1, 'rb') as file:
#     datas=pickle.load(file)
#
# print(datas)
# #print(datas.shape)
# new = datas['the_data']
# print(new)
# new1=new[0.0]
# #new3 = pd.DataFrame.from_dict(new1)
# print(new1)
# ry= new1[0]['t']
# print(ry)
# #x=new1['t']
# #print(x)
# my_dataframe = pd.DataFrame(ry)
# print(my_dataframe)
# ty=new1[0]['d']
# print(ty)
#
# new2=new[1.0]
# t2=new2[0]['d']
# new3=new[2.0]
# t3=new3[0]['d']
# new4=new[3.0]
# t4=new4[0]['d']
# new5=new[4.0]
# t5=new5[0]['d']
# new6=new[5.0]
# t6=new6[0]['d']
# new7=new[6.0]
# t7=new7[0]['d']
# new8=new[7.0]
# t8=new8[0]['d']
# new9=new[8.0]
# t9=new9[0]['d']
# new10=new[9.0]
# t10=new10[0]['d']
# new11=new[10.0]
# t11=new11[0]['d']
# new12=new[11.0]
# t12=new12[0]['d']
# new13=new[12.0]
# t13=new13[0]['d']
# new14=new[13.0]
# t14=new14[0]['d']
# new15=new[14.0]
# t15=new15[0]['d']
# new16=new[15.0]
# t16=new16[0]['d']
# new17=new[16.0]
# t17=new17[0]['d']
# new18=new[17.0]
# t18=new18[0]['d']
# new19=new[18.0]
# t19=new19[0]['d']
# new20=new[19.0]
# t20=new20[0]['d']
# new21=new[20.0]
# t21=new21[0]['d']
# new22=new[21.0]
# t22=new22[0]['d']
# new23=new[22.0]
# t23=new23[0]['d']
# new24=new[23.0]
# t24=new24[0]['d']
# new25=new[24.0]
# t25=new25[0]['d']
# new26=new[25.0]
# t26=new26[0]['d']
# new27=new[26.0]
# t27=new27[0]['d']
# new28=new[27.0]
# t28=new28[0]['d']
# new29=new[28.0]
# t29=new29[0]['d']
# new30=new[29.0]
# t30=new30[0]['d']
# result = np.concatenate((ty,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30))
# print(result)
# res=pd.DataFrame(result, columns = ['A', 'B', 'C','D','E'])
# print(res)
# X_dat=res[['A','B','C','D']]






import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
#shap.initjs()
#new = pd.DataFrame.from_dict(data)

file_path = '/home/leah/Downloads/Deeplearn_data_global.pkl'

# Open the pickle file for reading
with open(file_path, 'rb') as file:
    # Load the data from the pickle file
    dta = pickle.load(file)
print(dta)


# x1=dta[0][0.0]
# print(x1)
y=dta['y']
x=dta[['x1','x2', 'x3']]
model = xgb.XGBRegressor(objective= "reg:squarederror")
model.fit(x,y)
explainer=shap.Explainer(model)
print(explainer)
shap_values=explainer(x)
print(shap_values)
np.shape(shap_values.values)
# Create a waterfall plot
#shap.plots.waterfall(shap_values[0])
#shap.plots.waterfall(shap_values[1])
#shap.plots.waterfall(shap_values[2])


plt.figure()
plt.subplot(1,2,1)
shap.plots.waterfall(
    shap_values[0], show=False)
plt.subplot(1,2,2)
shap.plots.waterfall(
    shap_values[1], show=False)



# path1= '/home/leah/projects/KnowIt/leahs-synth-demo/leahs_mlp/bestmodel-epoch=7-valid_loss=0.00/archive/data.pkl'
# with open(path1, 'rb') as file:
#     object=pickle.load(file)
# print(object)



#shap_values= explainer()
#df = pd.DataFrame([data])
#print(df)
x=np.linspace(1,3,3)
y1=[0.4,0.6,0.7]
y2=[0.6,0.8,0.9]
y3=[0.1,0.5,0.6]
PDP=[0.43,0.63,0.73]

fig, ax20 = plt.subplots()
ax20.plot(x,y1,color="black", linestyle= '--', marker='o')
ax20.plot(x,y2,color="black", linestyle= '-.', marker='o')
ax20.plot(x,y3,color="black", linestyle= ':', marker='o')
ax20.plot(x,PDP,color="yellow")
plt.text(1,0.4,'i=1')
plt.text(2,0.6,'i=1')
plt.text(3,0.7,'i=1')


plt.text(1,0.6,'i=2')
plt.text(2,0.8,'i=2')
plt.text(3,0.9,'i=2')


plt.text(1,0.1,'i=3')
plt.text(2,0.5,'i=3')
plt.text(3,0.6,'i=3')
ax20.set_ylabel(r'$\^f_s$')
ax20.set_xlabel('Number of observations')
plt.show()
# res=data['results']
# print(res)
# att=res[2923]
# print(att)
# last=att[(0,0)]
# print(last)
# within=last['attributions']
# print(within)



x= [[-1.9823e-02, -4.6433e-02, -3.5809e-02, -1.2469e-02],
        [-1.9962e-02, -3.6710e-02, -1.0312e-01,  4.7214e-03],
        [-1.8299e-02, -3.8295e-02, -2.0489e-02,  2.8094e-04],
        [-2.1214e-02, -3.2716e-01, -1.8155e-02,  2.3168e-05],
        [-2.7828e-02, -2.8243e-02, -9.0340e-03,  1.2185e-02],
        [-1.3241e-01, -1.6061e-02, -4.3396e-03, -4.9330e-03],
        [-2.4111e-02, -1.6075e-02, -6.4965e-03, -4.1554e-03],
        [-1.8900e-02, -1.3745e-02,  1.6986e-03,  7.4371e-04],
        [-1.0523e-02, -8.6838e-03,  1.1933e-03,  6.3727e-03],
        [-1.4640e-02, -4.9247e-03,  2.0039e-03, -1.8333e-03],
        [-1.1908e-02,  7.0504e-03,  3.2147e-03,  3.9074e-03]]

# delete=x[0]
# print(delete)
# shap_values=explainer(x)
#shap.plots.waterfall(x[0])
ans = pd.DataFrame(x)
print(ans)



calc=ans[3]
print(calc)
x = np.arange(0, 11, 1)
x1= x.reshape(1,-1)
print(x1)
dat = np.array(ans[1])
print(dat)
dat_2 = dat.reshape(1, -1)
print(dat_2)

x_axis=np.arange(-5,6,1)
fig,ax5=plt.subplots(4,1,sharex=True)
fig.subplots_adjust(hspace=0)

ax5[0].plot(x_axis,ans[0])
ax5[1].plot(x_axis,ans[1])
ax5[2].plot(x_axis,ans[2])
ax5[3].plot(x_axis,ans[3])
ax5[0].grid(True)
ax5[1].grid(True)
ax5[2].grid(True)
ax5[3].grid(True)
ax5[0].set_ylabel('x1')
ax5[1].set_ylabel('x2')
ax5[2].set_ylabel('x3')
ax5[3].set_ylabel('x4')
ax5[3].set_xlabel('Time')

plt.show()

# x_axis=np.arange(-5,6,1)
# fig,ax6=plt.subplots(4,1,sharex=True)
# fig.subplots_adjust(hspace=0)
#
# ax6[0].scatter(x_axis,ans[0])
# ax6[1].scatter(x_axis,ans[1])
# ax6[2].scatter(x_axis,ans[2])
# ax6[3].scatter(x_axis,ans[3])
# ax6[0].grid(True)
# ax6[1].grid(True)
# ax6[2].grid(True)
# ax6[3].grid(True)
# plt.show()


x_axis = np.arange(-5, 6, 1)
fig, ax6 = plt.subplots(4, 1, sharex=True)
fig.subplots_adjust(hspace=0)

# Create some example data for the colors
colors = np.random.rand(len(x_axis))

# Plot each scatter plot with different colors
for i in range(4):
    sc = ax6[i].scatter(x_axis, ans[i], c=ans[i], cmap='viridis')
    ax6[i].grid(True)

# Add a color bar
ax6[0].set_ylabel('x1')
ax6[1].set_ylabel('x2')
ax6[2].set_ylabel('x3')
ax6[3].set_ylabel('x4')
ax6[3].set_xlabel('Time')
cbar = plt.colorbar(sc, ax=ax6)
cbar.set_label('Color Bar Label')
plt.show()

###################################################################################################################################################

# # Example 1D data
#
# data = np.array(ans[0])
#
# # Reshape the 1D data into a 2D array
# data_2d = data.reshape(1, -1)  # Reshape into a single row, -1 automatically calculates the number of columns
#
# # Display the 2D array as an image
# plt.imshow(data_2d, aspect='auto', cmap='viridis')  # Adjust aspect ratio and colormap as needed
# plt.colorbar()  # Add a colorbar to show the data scale
# plt.show()
#
# fig, ax9 = plt.subplots()
# x = np.arange(0, 11, 1)
# dat = np.array(ans[1])
#
# # Create 2D coordinate matrices using meshgrid
# #X, Y = np.meshgrid(x, np.arange(len(dat)))
#
# # Plot the data using pcolormesh
# pcm = ax9.pcolormesh(dat.reshape(1, -1), cmap='viridis')
#
# # Add colorbar
# plt.colorbar(pcm, ax=ax9)
#
# plt.show()
######################################################################################################################################################
#Next Trial

# fig,ax10 = plt.subplots(4,1)
# for i in range(4):
#         x = np.arange(0, 11, 1)
#         # x1 = x.reshape(1, -1)
#         dat = np.array(ans[i])
#         dat_2 = dat.reshape(1, -1)
#         ax10[i].plot(x, dat, color='blue')
#
#         sc = ax10[i].pcolormesh(dat_2, cmap='viridis')
#
#
# #        veg=['ans0', 'ans1', 'ans2', 'ans3']
# #        ax10.set_yticks(np.arange(len(veg)), labels=veg)
# plt.show()

# fig, ax11 = plt.subplots()
# x= np.arange(1, 12, 1)
# dat = np.array(ans[3])
# ax11.plot(x, dat, color='blue')
#
# y=np.arange(-0.05, 0.05, 0.10)
# dat_2 = dat.reshape(1, -1)
# X, Y = np.meshgrid(x, y)
# ax11.pcolormesh(x, y, dat_2, cmap='viridis')
# plt.show()




# fig, ax11 = plt.subplots()
#
# # Generate x values
# x = np.arange(1, 12, 1)
#
# # Generate y values
# y = np.linspace(-0.05, 0.05, len(x))
#
# # Generate data (example)
# dat = np.array(ans[3])
#
# # Plot line plot
# ax11.plot(x, dat, color='blue')
#
# # Create meshgrid for pcolormesh
# X, Y = np.meshgrid(x, y)
#
# # Repeat dat along the y-axis to match the shape of X and Y
# dat_2 = np.repeat(dat.reshape(1, -1), len(y), axis=0)
#
# # Plot pcolormesh
# pc = ax11.pcolormesh(X, Y, dat_2, cmap='viridis')
#
# # Add colorbar
# plt.colorbar(pc, ax=ax11)
#
# plt.show()


fig, ax12 = plt.subplots(4,1, sharex=True)
fig.suptitle('A combination of a line plot and a scatterplot ')
# Generate x values
for i in range(4):

    x = np.arange(1, 12, 1)

# Generate y values
    y = np.linspace(min(ans[i]) - 0.01, max(ans[i]) + 0.01, len(x))

# Generate data (example)
    dat = np.array(ans[i])

# Plot line plot
    ax12[i].plot(x, dat, color='blue')

# Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x, y)

# Repeat dat along the y-axis to match the shape of X and Y
    dat_2 = np.repeat(dat.reshape(1, -1), len(y), axis=0)

# Plot pcolormesh
    pc = ax12[i].pcolormesh(X, Y, dat_2, cmap='viridis')
ax12[0].set_ylabel('x1')
ax12[1].set_ylabel('x2')
ax12[2].set_ylabel('x3')
ax12[3].set_ylabel('x4')
ax12[3].set_xlabel('Time')

# Add colorbar
plt.colorbar(pc, ax=ax12)

plt.show()




fig,ax5 = plt.subplots(4,1)
for i in range(4):
        dat= np.array(ans[i])
        dat_2= dat.reshape(1,-1)
        sc = ax5[i].imshow(dat_2, cmap='viridis')

#        x = np.arange(0,12,1)  # Assuming x-axis values are just indices
#        y = dat_2
#        ax5[i].plot(x,y, color='red')
ax5[0].set_ylabel('x1')
ax5[1].set_ylabel('x2')
ax5[2].set_ylabel('x3')
ax5[3].set_ylabel('x4')
ax5[3].set_xlabel('Time')
plt.colorbar(sc, ax=ax5)
plt.show()



fig, ax15 = plt.subplots()
veg = ["x1", "x2", "x3", "x4"]
sc = ax15.imshow(result, cmap='viridis')
ax15.set_yticks(np.arange(len(veg)))
ax15.set_yticklabels(veg)
ax15.set_xlabel('Time')
ax15.set_ylabel('Features')
plt.colorbar(sc, ax=ax15)
fig.tight_layout()
plt.show()



fig, ax5 = plt.subplots(4, 1, figsize=(8, 6))  # Adjust figsize as needed
for i in range(4):
    x = np.arange(0, 11, 1)
    x1 = x.reshape(1, -1)
    dat = np.array(ans[i])
    dat_2 = dat.reshape(1, -1)
#    y = dat_2.squeeze()  # Remove unnecessary dimensions

    X, Y = np.meshgrid(x1, dat_2)
    sc = ax5[i].pcolormesh(X, Y, dat_2, cmap='viridis')

plt.show()

fig, ax5 = plt.subplots(4, 1,figsize=(8,6))  # Adjust figsize as needed
for i in range(4):
        x = np.arange(0, 11, 1)
        x1= x.reshape(1,-1)
        dat = np.array(ans[i])
        dat_2 = dat.reshape(1, -1)
        y = dat_2.squeeze()  # Remove unnecessary dimensions
        ax5[i].plot(x1, dat_2, color='blue')  # Overlay line plot

        a=dat_2
        X,Y=np.meshgrid(x1,a)
        sc = ax5[i].pcolormesh(X,Y,y)

    # Assuming x-axis values are just indices from 0 to 11


# Add colorbar to the last axis
#plt.colorbar(sc, ax=ax5[-1])

#plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()


#print(new)
#DeepLiftShap-eval-success-100-True-(2923, 3023)
#print(y)

dat1 = pd.read_pickle( '/home/leah/Downloads/LEAH_SYNTH_DATA/synth_1/synth_1.pickle')
print(dat1)


plt.ioff()
matplotlib.is_interactive()
data= pd.read_pickle('/home/leah/leah_visualisation/Pickle Data/RandomSampling_shifted_10000_use.pkl')
print(data)

data.index.name = 'DateTime'
data1=data.drop(["y_shift","y"],axis='columns')
data1['Timestamp'] = data1.index


Timestamp=data1.Timestamp
rs=data1.reset_index(drop=True)

dat3 = pd.read_pickle('/home/leah/leah_visualisation/Pickle Data/Deeplearn_data_global.pkl')
print(dat3)
fig,ax1=plt.subplots()
x=list(range(0,714))
ax1.plot(x,dat3['y'])
ax1.grid(True)
plt.xticks(rotation=25)
plt.show()


fig,ax2=plt.subplots()
x=list(range(0,714))
cmap = 'viridis'
r=ax2.scatter(x, dat3['y'], c=dat3['y'], cmap=cmap)
#ax2.scatter(x,dat3['y'] )
fig.colorbar(r)
ax2.grid(True)
plt.xticks(rotation=25)
plt.show()


#import matplotlib.pyplot as plt
#import numpy as np

# Generate sample data
#x = np.random.rand(100)
#y = np.random.rand(100)
#ata_values = np.random.rand(100)  # Example data values for color mapping

# Define colormap and normalization
cmap = 'viridis'  # Choose a colormap
norm = plt.Normalize(vmin=min(dat3['y']), vmax=max(dat3['y']))  # Normalize data values
# Create scatterplot with color mapping
plt.scatter(x, dat3['y'], c=dat3['y'] ,cmap=cmap, norm=norm)
# Add colorbar
plt.colorbar(label='Data Values')
# Add labels and title
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatterplot with Color Mapping')
# Show plot
plt.show()

#Multiple line graphs
x=list(range(0,714))
fig,ax3=plt.subplots(3,1,sharex=True)
fig.subplots_adjust(hspace=0)

ax3[0].plot(x,dat3['x1_att'])
ax3[1].plot(x,dat3['x2_att'])
ax3[2].plot(x,dat3['x3_att'])
ax3[0].grid(True)
ax3[1].grid(True)
ax3[2].grid(True)
plt.show()

x=list(range(0,714))
fig,ax3=plt.subplots(3,1,sharex=True)
fig.subplots_adjust(hspace=0)

ax3[0].scatter(x,dat3['x1_att'])
ax3[1].scatter(x,dat3['x2_att'])
ax3[2].scatter(x,dat3['x3_att'])
ax3[0].grid(True)
ax3[1].grid(True)
ax3[2].grid(True)

plt.show()

###############################################################################################################################################


# Create an explainer object with the loaded model
#explainer = shap.Explainer(model)

# Compute SHAP values for a set of instances
#shap_values = explainer.shap_values(X)

# Visualize SHAP values
#shap.summary_plot(shap_values, X)
