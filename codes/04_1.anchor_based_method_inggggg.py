
# 밝기 보정 및 pastel 처리
def lift_brightness(rgb, min_lightness=0.25):
    out = []
    for r, g, b in rgb:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        if l < min_lightness:
            rgb_bright = colorsys.hls_to_rgb(h, min_lightness, s)
            out.append(rgb_bright)
        else:
            out.append((r, g, b))
    return np.array(out)


def mix_with_white(rgb, w=0.3):
    white = np.ones(3)
    return rgb * (1 - w) + white * w


anchor_list = list(anchor_rgb_genes.keys())
anchor_colors = np.array([anchor_rgb_genes[k] for k in anchor_list])  # (3, 3)
co_matrix_arr = co_matrix_df[anchor_list].values  # (714, 3)
rgb_arr = np.dot(co_matrix_arr, anchor_colors)    # (714, 3)

min_lightness = 0.3
rgb_arr = lift_brightness(rgb_arr, min_lightness)
rgb_arr = mix_with_white(rgb_arr, w=0.3)


# Fmr1로 어둡게... but do we need this? 
darkening_gene = 'Fmr1'

dark_arr = co_matrix_df[darkening_gene].values.reshape(-1, 1)  # (n_gene, 1)
darken_strength = 0.1
rgb_arr = rgb_arr * (1 - darken_strength * dark_arr)

L2_rgb_dict_df = pd.DataFrame(rgb_arr, columns=['R', 'G', 'B'], index=co_matrix_df.index)
L2_rgb_dict_df['hex_val'] = L2_rgb_dict_df.apply(lambda x: to_hex([x['R'], x['G'], x['B']]), axis=1)
L2_rgb_dict_df['gene'] = L2_rgb_dict_df.index
plot_3d_rgb(L2_rgb_dict_df, 'hex_val')




tops_R = list(co_matrix_df.sort_values('Arc', ascending = False).iloc[0:20].index)
tops_G = list(co_matrix_df.sort_values('Nsf', ascending = False).iloc[0:20].index)
tops_B = list(co_matrix_df.sort_values('Prkcz', ascending = False).iloc[0:20].index)

rgb_to_save = L2_rgb_dict_df[['hex_val','gene','candy']]

rgb_to_save['top'] = ['R' if a in tops_R else 'G' if a in tops_G else 'B' if a in tops_B else 'X' for a in list(rgb_to_save.index)]

rgb_to_save.to_csv(datapath+'04.L2_rgb_dict_df.csv')




















binary_weight = 0.9  # 한 번이라도 들어간 효과를 얼마나 강하게? (0.5~0.8 사이 추천)
ratio_weight = 1 - binary_weight

anchor_binary = (co_matrix_arr > 0).astype(float)
binary_sum = anchor_binary.sum(axis=1, keepdims=True)
binary_sum[binary_sum == 0] = 1
anchor_binary_rgb = np.dot(anchor_binary / binary_sum, anchor_colors)

anchor_sum = co_matrix_arr.sum(axis=1, keepdims=True)
anchor_sum[anchor_sum == 0] = 1e-8
anchor_ratio = co_matrix_arr / anchor_sum
anchor_ratio_rgb = np.dot(anchor_ratio, anchor_colors)

# blend
anchor_rgb = anchor_binary_rgb * binary_weight + anchor_ratio_rgb * ratio_weight

# white와 혼합(비율은 실제 anchor 합, 또는 원하는 값으로 조절)
total_anchor = co_matrix_arr.sum(axis=1, keepdims=True)
total_anchor = np.clip(total_anchor, 0, 1)
rgb_arr = (1 - total_anchor) * np.ones(3) + total_anchor * anchor_rgb

# Fmr1이 1이면 black
fmr1_mask = (co_matrix_df[darkening_gene] == 1).values
rgb_arr[fmr1_mask, :] = 0.0

# pastel 느낌 원하면 mix_with_white 추가
def mix_with_white(rgb, w=0.15):
    white = np.ones(3)
    return rgb * (1 - w) + white * w

rgb_arr = mix_with_white(rgb_arr, w=0.10)

L2_rgb_dict_df = pd.DataFrame(rgb_arr, columns=['R', 'G', 'B'], index=co_matrix_df.index)
L2_rgb_dict_df['hex_val'] = L2_rgb_dict_df.apply(lambda x: to_hex([x['R'], x['G'], x['B']]), axis=1)
L2_rgb_dict_df['gene'] = L2_rgb_dict_df.index

plot_3d_rgb(L2_rgb_dict_df)






node_rgb = {}
for gene in list(co_matrix_df.index):
    rgb = np.zeros(3)
    # add color according to anchor 
    for anchor, color in anchor_rgb_genes.items():
        w = co_matrix_df.loc[gene][anchor] 
        rgb += w * np.array(color)
    rgb = np.clip(rgb, 0, 1)
    # darkening with fmr1 
    w_dark = co_matrix_df.loc[gene][darkening_gene]
    rgb = rgb * (1 - 0.2 * w_dark)  # 밝기 조절
    node_rgb[gene] = rgb


L2_rgb_dict_df = pd.DataFrame(node_rgb).T
L2_rgb_dict_df.columns = ['R','G','B']


L2_hex_val = [to_hex(rgb) for gene, rgb in node_rgb.items()]
L2_hls_val = [to_hex(brighten_by_hls(rgb),0.1) for gene, rgb in node_rgb.items()]


L2_rgb_dict_df['hex_val'] = L2_hex_val
L2_rgb_dict_df['hls_val'] = L2_hls_val
L2_rgb_dict_df['gene'] = list(L2_rgb_dict_df.index)
L2_rgb_dict_df.loc[candidategenes]

plot_3d_rgb(L2_rgb_dict_df, 'hls_val')




tops_R = list(co_matrix_df.sort_values('Arc', ascending = False).iloc[0:20].index)
tops_G = list(co_matrix_df.sort_values('Nsf', ascending = False).iloc[0:20].index)
tops_B = list(co_matrix_df.sort_values('Prkcz', ascending = False).iloc[0:20].index)

rgb_to_save = L2_rgb_dict_df[['hex_val','hls_val','gene','candy']]

rgb_to_save['top'] = ['R' if a in tops_R else 'G' if a in tops_G else 'B' if a in tops_B else 'X' for a in list(rgb_to_save.index)]

rgb_to_save.to_csv(datapath+'04.L2_rgb_dict_df.csv')







# 이거 진짜 마무리 하고 싶었는데 
아냐 다섯시에 가자 
이거 진짜 끝내자 






# Step 2: anchor gene 설정 + RGB 색상 지정 (Fmr1은 따로 다룸)

node_rgb = {}
for gene in list(co_matrix_df.index):
    rgb = np.zeros(3)
    # 색상 anchor 섞기
    for anchor, color in anchor_rgb_genes.items():
        w = co_matrix_df.loc[gene][anchor]
        rgb += w * np.array(color)
    rgb = np.clip(rgb, 0, 1)
    # Fmr1 비율 기반으로 밝기 줄이기
    w_dark = co_matrix_df.loc[gene][darkening_gene]
    rgb = rgb * (1 - w_dark)  # 밝기 조절
    node_rgb[gene] = rgb


# Step 5: RGB → hex
node_hex = {gene: to_hex(rgb) for gene, rgb in node_rgb.items()}


# Step 6: 밝기 조정 (HLS)
def brighten_by_hls(rgb, min_lightness=0.7):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(l, min_lightness)
    return colorsys.hls_to_rgb(h, l, s)


node_hls = {gene: to_hex(brighten_by_hls(rgb)) for gene, rgb in node_rgb.items()}


# Step 6: 결과 저장
df_out = pd.DataFrame({
    "gene": list(co_matrix_df.index),
    "hex_val": [node_hex[g] for g in list(co_matrix_df.index)],
    "hls_val": [node_hls[g] for g in list(co_matrix_df.index)],
    "R": [co_matrix_df.loc[g]['Arc'] for g in list(co_matrix_df.index)],
    "B": [co_matrix_df.loc[g]['Prkcz'] for g in list(co_matrix_df.index)],
    "G": [co_matrix_df.loc[g]['Nsf'] for g in list(co_matrix_df.index)],
    "Fmr1": [co_matrix_df.loc[g]['Fmr1'] for g in list(co_matrix_df.index)],
})



plot_3d_rgb(df_out, 'hls_val')









# 그럼 그냥 3개로 정해놓고 하면 ??? 
# 그러면 그냥 색이 덜한 애들도 있는거임. 


# Step 4: RGB 섞기
node_rgb = {}
for gene in list(co_matrix_df.index):
    rgb = np.zeros(3)
    for anchor, anchor_rgb in anchor_rgb_genes.items():
        w = co_matrix_df.loc[gene][anchor]
        rgb += w * np.array(anchor_rgb)
    rgb = np.clip(rgb, 0, 1)  # This ensures that the RGB values are within the valid range [0, 1]
    node_rgb[gene] = rgb

# Step 5: RGB → hex
node_hex = {gene: to_hex(rgb) for gene, rgb in node_rgb.items()}


# Step 6: 밝기 조정 (HLS)
def brighten_by_hls(rgb, min_lightness=0.7):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(l, min_lightness)
    return colorsys.hls_to_rgb(h, l, s)

node_hls = {gene: to_hex(brighten_by_hls(rgb)) for gene, rgb in node_rgb.items()}


# Step 7: 결과 저장
df_out = pd.DataFrame({
    "gene": list(co_matrix_df.index),
    "hex_val": [node_hex[g] for g in list(co_matrix_df.index)],
    "hls_val": [node_hls[g] for g in list(co_matrix_df.index)],
    "R": [co_matrix_df.loc[g]['Arc'] for g in list(co_matrix_df.index)],
    "B": [co_matrix_df.loc[g]['Prkcz'] for g in list(co_matrix_df.index)],
    "G": [co_matrix_df.loc[g]['Nsf'] for g in list(co_matrix_df.index)]
})

df_out[df_out.gene.isin(candidategenes)]
plot_3d_rgb(df_out, 'hls_val')


