import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D

cind = np.array([[0.549941522430559, 0.5701589629647961, 0.5750277918844569, 0.5076087036181521, 0.5822519225641558, 0.5526502499503709, 0.6512520330813908, 0.6298432826994217], [0.5387140049827767, 0.5504567784594284, 0.5540639158209056, 0.5096161302482466, 0.5667747808206022, 0.5411003295651928, 0.6204834188457167, 0.618051369493592], [0.5305539322353077, 0.5443049907624053, 0.5479088889558621, 0.5048990694359066, 0.5626359046333388, 0.5334434110319942, 0.6133710535854328, 0.6165719390283995], [0.517827252049482, 0.5299624678379438, 0.5343675542856691, 0.50918809556289, 0.5518255060664347, 0.5187275185573517, 0.5877870223922262, 0.6072045343301934], [0.5089131240673521, 0.5222906528890879, 0.5250614187536259, 0.49020644494194454, 0.5374382628860203, 0.5104959247356154, 0.571716100792625, 0.5884576602641837], [0.5244759598135058, 0.5351875245950369, 0.5383192136995983, 0.5081814405680586, 0.5363250734060783, 0.5255245133957231, 0.5705285148690147, 0.6020132666115197], [0.5091861210018432, 0.5187108617246011, 0.5204498691023103, 0.5007089771333254, 0.5345311923308791, 0.5105401632441459, 0.5719976614692222, 0.5918027545987329], [0.5150199564949657, 0.5258871286719204, 0.5271349204040394, 0.49640751626897217, 0.5389908583550023, 0.5161186835715134, 0.5529820335798351, 0.5955412105815113]])
auc = np.array([[0.551561500968786, 0.5736554207969191, 0.5789487601620542, 0.5045127420085064, 0.5943678823706987, 0.5570282895593394, 0.668416740089424, 0.6408502383947815], [0.5417010524086562, 0.5553658321246515, 0.5598160787183083, 0.5102646878707218, 0.5781257689976029, 0.5465052179514217, 0.6443340554084002, 0.6292298573140878], [0.533015749031936, 0.5499923906067044, 0.5546712700715969, 0.5027335675957284, 0.5665118745208393, 0.5352917866428741, 0.6286887137023762, 0.6226880050551397], [0.5220978230440824, 0.5349949156046816, 0.5394805310266935, 0.5130598307643439, 0.5612354827433896, 0.5190221604116877, 0.6076584558465378, 0.6178249804277096], [0.5048419306647758, 0.5205144393584609, 0.5237185942224566, 0.48747150325358396, 0.5492440587747414, 0.5121298189364673, 0.5860140280656786, 0.5985645103420665], [0.530329155662711, 0.5418346678846313, 0.546370863975199, 0.5106561074358547, 0.5498813239467281, 0.5344823950549239, 0.5815164713469585, 0.613804447012974], [0.5080592176391125, 0.5182484331997522, 0.5204021077204325, 0.4987814213261796, 0.5408843892942731, 0.5131576959731882, 0.581073813712402, 0.5982094003482722], [0.5144401874683059, 0.5280778926911136, 0.5302108802617698, 0.4951332932722953, 0.5397195367708062, 0.5175431067276836, 0.5617482758857159, 0.6048619861187781]])

cind=cind[:8]
auc=auc[:8]


# Define the distributional shift intensities
itys = [1, 2, 3, 4, 5, 6, 7, 8]

# Set Seaborn style
sns.set(style="whitegrid", palette="deep")

# Create the subplots in a single row with 4 columns
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# Custom line styles for legend
custom_lines = [
    Line2D([0], [0], color='dodgerblue', lw=2),  # Regular Cox
    Line2D([0], [0], color='orange', lw=2),      # Ridge Cox
    Line2D([0], [0], color='green', lw=2),       # Lasso Cox
    Line2D([0], [0], color='brown', lw=2),       # Elastic Net Cox
    Line2D([0], [0], color='purple', lw=2),      # Sample Splitting Cox
    Line2D([0], [0], color='pink', lw=2),        # AFT
    Line2D([0], [0], color='grey', lw=2),  # RSF
    Line2D([0], [0], color='red', lw=2)          # DRL Cox
]

# C-index plot
axes[0].plot(itys, cind[:, 0], label='Regular Cox', color='dodgerblue', linewidth=2)
axes[0].plot(itys, cind[:, 1], label='Ridge Cox', color='orange', linewidth=2)
axes[0].plot(itys, cind[:, 2], label='Lasso Cox', color='green', linewidth=2)
axes[0].plot(itys, cind[:, 3], label='Elastic Net Cox', color='brown', linewidth=2)
axes[0].plot(itys, cind[:, 4], label='Sample Splitting Cox', color='purple', linewidth=2)
axes[0].plot(itys, cind[:, 5], label='AFT', color='pink', linewidth=2)
axes[0].plot(itys, cind[:, 6], label='RSF', color='grey', linewidth=2)
axes[0].plot(itys, cind[:, 7], label='DRL Cox', color='red', linewidth=2)
axes[0].set_xlabel("Shift Intensity", fontsize=14)
axes[0].set_ylabel("C-index", fontsize=14)
axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# Increase of C-index plot
axes[1].plot(itys, cind[:, 1] - cind[:, 0], label='Ridge Cox', color='orange', linewidth=2)
axes[1].plot(itys, cind[:, 2] - cind[:, 0], label='Lasso Cox', color='green', linewidth=2)
axes[1].plot(itys, cind[:, 3] - cind[:, 0], label='Elastic Net Cox', color='brown', linewidth=2)
axes[1].plot(itys, cind[:, 4] - cind[:, 0], label='Sample Splitting Cox', color='purple', linewidth=2)
axes[1].plot(itys, cind[:, 5] - cind[:, 0], label='AFT', color='pink', linewidth=2)
axes[1].plot(itys, cind[:, 6] - cind[:, 0], label='RSF', color='grey', linewidth=2)
axes[1].plot(itys, cind[:, 7] - cind[:, 0], label='DRL Cox', color='red', linewidth=2)
axes[1].set_xlabel("Shift Intensity", fontsize=14)
axes[1].set_ylabel("Increase of C-index", fontsize=14)
axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# iAUC plot
axes[2].plot(itys, auc[:, 0], label='Regular Cox', color='dodgerblue', linewidth=2)
axes[2].plot(itys, auc[:, 1], label='Ridge Cox', color='orange', linewidth=2)
axes[2].plot(itys, auc[:, 2], label='Lasso Cox', color='green', linewidth=2)
axes[2].plot(itys, auc[:, 3], label='Elastic Net Cox', color='brown', linewidth=2)
axes[2].plot(itys, auc[:, 4], label='Sample Splitting Cox', color='purple', linewidth=2)
axes[2].plot(itys, auc[:, 5], label='AFT', color='pink', linewidth=2)
axes[2].plot(itys, auc[:, 6], label='RSF', color='grey', linewidth=2)
axes[2].plot(itys, auc[:, 7], label='DRL Cox', color='red', linewidth=2)
axes[2].set_xlabel("Shift Intensity", fontsize=14)
axes[2].set_ylabel("iAUC", fontsize=14)
axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# Increase of iAUC plot
axes[3].plot(itys, auc[:, 1] - auc[:, 0], label='Ridge Cox', color='orange', linewidth=2)
axes[3].plot(itys, auc[:, 2] - auc[:, 0], label='Lasso Cox', color='green', linewidth=2)
axes[3].plot(itys, auc[:, 3] - auc[:, 0], label='Elastic Net Cox', color='brown', linewidth=2)
axes[3].plot(itys, auc[:, 4] - auc[:, 0], label='Sample Splitting Cox', color='purple', linewidth=2)
axes[3].plot(itys, auc[:, 5] - auc[:, 0], label='AFT', color='pink', linewidth=2)
axes[3].plot(itys, auc[:, 6] - auc[:, 0], label='RSF', color='grey', linewidth=2)
axes[3].plot(itys, auc[:, 7] - auc[:, 0], label='DRL Cox', color='red', linewidth=2)
axes[3].set_xlabel("Shift Intensity", fontsize=14)
axes[3].set_ylabel("Increase of iAUC", fontsize=14)
axes[3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[3].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# Adjust layout to ensure plots are properly spaced
plt.tight_layout()

# Add a single legend at the bottom center, further down to avoid overlap
fig.legend(custom_lines, ['Regular Cox', 'Ridge Cox', 'Lasso Cox', 'Elastic Net Cox', 
                          'Sample Splitting Cox', 'AFT', 'RSF', 'DRL Cox'], loc='center', bbox_to_anchor=(0.5, -0.2), 
           ncol=4, fontsize=20)
fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] * 0.7)
# Save the final combined plot
plt.savefig("whas500", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
