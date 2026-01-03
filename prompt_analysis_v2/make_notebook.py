# make_notebook.py
import json
from pathlib import Path

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prompt Trajectories + Specificity Visualizations\n",
    "\n",
    "Reads `data/prompt_trajectories.csv` + `data/session_summary.csv` and plots:\n",
    "- heatmap of intent over time\n",
    "- specificity trajectory\n",
    "- constraint trajectory\n",
    "- distributions\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "traj = pd.read_csv('../data/prompt_trajectories.csv')\n",
    "summary = pd.read_csv('../data/session_summary.csv')\n",
    "traj.head(), summary.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1) Intent heatmap\n",
    "Each row = session, columns = normalized timestep bins. Values = most common intent in that bin.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "intent_to_id = {k:i for i,k in enumerate(sorted(traj['prompt_intent'].unique()))}\n",
    "id_to_intent = {v:k for k,v in intent_to_id.items()}\n",
    "\n",
    "# bin by t_norm into 20 bins\n",
    "bins = 20\n",
    "traj['bin'] = (traj['t_norm'] * (bins-1)).round().astype(int)\n",
    "\n",
    "sessions = sorted(traj['conversation_id'].unique())\n",
    "mat = np.full((len(sessions), bins), np.nan)\n",
    "\n",
    "for r, sid in enumerate(sessions):\n",
    "    sub = traj[traj['conversation_id']==sid]\n",
    "    for b in range(bins):\n",
    "        subb = sub[sub['bin']==b]\n",
    "        if len(subb)==0:\n",
    "            continue\n",
    "        # majority intent\n",
    "        top_intent = subb['prompt_intent'].value_counts().idxmax()\n",
    "        mat[r,b] = intent_to_id[top_intent]\n",
    "\n",
    "plt.figure(figsize=(12, max(3, len(sessions)*0.4)))\n",
    "im = plt.imshow(mat, aspect='auto')\n",
    "cbar = plt.colorbar(im)\n",
    "cbar.set_ticks(list(id_to_intent.keys()))\n",
    "cbar.set_ticklabels([id_to_intent[i] for i in id_to_intent.keys()])\n",
    "plt.yticks(range(len(sessions)), sessions)\n",
    "plt.xticks(range(bins), [f\"{i/(bins-1):.2f}\" for i in range(bins)], rotation=90)\n",
    "plt.title('Intent heatmap over normalized time')\n",
    "plt.xlabel('Normalized timestep (binned)')\n",
    "plt.ylabel('Session')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2) Specificity trajectory\n",
    "`specificity_pred` is the sklearn score (continuous). If it’s missing, run `learn_prompt_specificity.py`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "for sid in sessions:\n",
    "    sub = traj[traj['conversation_id']==sid].sort_values('timestep')\n",
    "    if sub['specificity_pred'].notna().any():\n",
    "        plt.plot(sub['t_norm'], sub['specificity_pred'], alpha=0.7, label=sid)\n",
    "plt.title('Specificity (sklearn) over time')\n",
    "plt.xlabel('t_norm')\n",
    "plt.ylabel('specificity_pred')\n",
    "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3) Constraints + descriptive words trajectory\n",
    "Rule-based counts (baseline)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "for sid in sessions:\n",
    "    sub = traj[traj['conversation_id']==sid].sort_values('timestep')\n",
    "    plt.plot(sub['t_norm'], sub['num_constraints_rule'], alpha=0.7, label=sid)\n",
    "plt.title('Rule-based constraint count over time')\n",
    "plt.xlabel('t_norm')\n",
    "plt.ylabel('num_constraints_rule')\n",
    "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "for sid in sessions:\n",
    "    sub = traj[traj['conversation_id']==sid].sort_values('timestep')\n",
    "    plt.plot(sub['t_norm'], sub['descriptive_words'], alpha=0.7, label=sid)\n",
    "plt.title('Descriptive word proxy over time')\n",
    "plt.xlabel('t_norm')\n",
    "plt.ylabel('descriptive_words')\n",
    "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4) Distributions\n",
    "Check whether the signals are too sparse."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8,4))\n",
    "plt.hist(traj['num_constraints_rule'], bins=20)\n",
    "plt.title('Distribution: num_constraints_rule')\n",
    "plt.xlabel('count')\n",
    "plt.ylabel('frequency')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "plt.hist(traj['prompt_length'], bins=20)\n",
    "plt.title('Distribution: prompt_length')\n",
    "plt.xlabel('words')\n",
    "plt.ylabel('frequency')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "if traj['specificity_pred'].notna().any():\n",
    "    plt.figure(figsize=(8,4))\n",
    "    plt.hist(traj['specificity_pred'].dropna(), bins=20)\n",
    "    plt.title('Distribution: specificity_pred')\n",
    "    plt.xlabel('score')\n",
    "    plt.ylabel('frequency')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/prompt_visualizations.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("✅ wrote notebooks/prompt_visualizations.ipynb")
