import numpy as np, pandas as pd, pickle
from sklearn.model_selection import KFold

# Phase 2 Execution: Build and save the clean datasets and stateful preprocessing artifacts
import pickle

def run_phase2_preprocessing(train_df, test_df, deduplicate_train=True):
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    
    # 1. Strip whitespace on all string columns
    str_cols = ['airline', 'flight', 'source_city', 'departure_time', 
                'stops', 'arrival_time', 'destination_city', 'flight_class']
    for col in str_cols:
        train_clean[col] = train_clean[col].astype(str).str.strip()
        test_clean[col] = test_clean[col].astype(str).str.strip()
        
    # 2. Remove exact feature + price duplicates in Train (6 rows where identical flight/features/price were logged twice)
    if deduplicate_train:
        before_len = len(train_clean)
        feature_and_target_cols = str_cols + ['duration', 'days_left', 'price']
        train_clean = train_clean.drop_duplicates(subset=feature_and_target_cols, keep='first').reset_index(drop=True)
        print(f"Removed {before_len - len(train_clean)} exact feature+price duplicate rows from Train. New Train shape: {train_clean.shape}")
        
    # 3. Store stateful preprocessing metadata (known categories, ordinal maps, numerical bounds) learned from Train
    preproc_state = {
        'str_cols': str_cols,
        'known_categories': {col: sorted(train_clean[col].unique().tolist()) for col in str_cols},
        'stops_map': {'zero': 0, 'one': 1, 'two_or_more': 2},
        'class_map': {'Economy': 0, 'Business': 1},
        'time_map': {
            'Early_Morning': 0,
            'Morning': 1,
            'Afternoon': 2,
            'Evening': 3,
            'Night': 4,
            'Late_Night': 5
        },
        'zero_price_flights': sorted(train_clean[train_clean['price'] == 0]['flight'].unique().tolist()),
        'train_price_min': float(train_clean['price'].min()),
        'train_price_max': float(train_clean['price'].max())
    }
    
    # Check unseen categories in Test against Train's known_categories
    for col in str_cols:
        unseen = set(test_clean[col].unique()) - set(preproc_state['known_categories'][col])
        preproc_state[f'unseen_in_test_{col}'] = sorted(list(unseen))
        
    return train_clean, test_clean, preproc_state




def build_deterministic_features(df, preproc_state):
    """
    Step 3A: Deterministic & Domain Feature Engineering (Row-level, zero target dependency)
    """
    d = df.copy()
    
    # 1. Flight Code & Number Extraction
    flight_split = d['flight'].str.split(' ', expand=True)
    d['flight_prefix'] = flight_split[0]
    d['flight_num'] = flight_split[1].astype(int)
    # Regional vs Mainline indicator (flight numbers >= 3000 are typically regional/SkyWest/operated-by flights)
    d['is_regional_flight_num'] = (d['flight_num'] >= 3000).astype(int)
    d['flight_num_tier'] = (d['flight_num'] // 1000).astype(int)
    
    # Flag if flight is one of the 15 known zero-price candidate flights from Train
    d['is_zero_price_candidate_flight'] = d['flight'].isin(preproc_state['zero_price_flights']).astype(int)
    
    # 2. Ordinal Encodings using stored mappings from Phase 2
    d['stops_num'] = d['stops'].map(preproc_state['stops_map']).fillna(-1).astype(int)
    d['class_num'] = d['flight_class'].map(preproc_state['class_map']).fillna(0).astype(int)
    d['dep_time_num'] = d['departure_time'].map(preproc_state['time_map']).fillna(-1).astype(int)
    d['arr_time_num'] = d['arrival_time'].map(preproc_state['time_map']).fillna(-1).astype(int)
    
    # Cyclical & Time-of-Day Features
    d['dep_sin'] = np.sin(2 * np.pi * d['dep_time_num'] / 6.0)
    d['dep_cos'] = np.cos(2 * np.pi * d['dep_time_num'] / 6.0)
    d['arr_sin'] = np.sin(2 * np.pi * d['arr_time_num'] / 6.0)
    d['arr_cos'] = np.cos(2 * np.pi * d['arr_time_num'] / 6.0)
    d['time_slot_diff'] = (d['arr_time_num'] - d['dep_time_num']) % 6
    d['is_late_night_dep'] = (d['departure_time'] == 'Late_Night').astype(int)
    d['is_peak_dep'] = d['departure_time'].isin(['Morning', 'Evening']).astype(int)
    d['is_red_eye_or_early'] = d['departure_time'].isin(['Early_Morning', 'Late_Night', 'Night']).astype(int)
    
    # 3. Route & Hub Geography Features (California Corridor: North vs South)
    # North CA: San Francisco, San Jose, Sacramento | South CA: Los Angeles, San Diego
    north_cities = {'San Francisco', 'San Jose', 'Sacramento'}
    d['src_is_north'] = d['source_city'].isin(north_cities).astype(int)
    d['dst_is_north'] = d['destination_city'].isin(north_cities).astype(int)
    d['is_intra_region'] = (d['src_is_north'] == d['dst_is_north']).astype(int) # e.g. San Jose <-> Sacramento or LAX <-> SAN
    
    # Approximate Great-Circle Air Distance (miles) between the 5 California airports
    coords = {
        'San Francisco': (37.6213, -122.3790),
        'San Jose': (37.3639, -121.9289),
        'Sacramento': (38.6951, -121.5908),
        'Los Angeles': (33.9416, -118.4085),
        'San Diego': (32.7338, -117.1933)
    }
    lat1 = d['source_city'].map(lambda x: coords[x][0])
    lon1 = d['source_city'].map(lambda x: coords[x][1])
    lat2 = d['destination_city'].map(lambda x: coords[x][0])
    lon2 = d['destination_city'].map(lambda x: coords[x][1])
    # Euclidean degree approximation scaled to statute miles
    d['route_distance_miles'] = np.sqrt(((lat2 - lat1) * 69.0)**2 + ((lon2 - lon1) * 55.0)**2).round(1)
    
    # Directed and Undirected Route Combinations
    d['route'] = d['source_city'] + '_' + d['destination_city']
    d['undirected_route'] = np.minimum(d['source_city'], d['destination_city']) + '_' + np.maximum(d['source_city'], d['destination_city'])
    d['airline_class'] = d['airline'] + '_' + d['flight_class']
    d['route_class'] = d['route'] + '_' + d['flight_class']
    d['route_airline_class'] = d['route'] + '_' + d['airline'] + '_' + d['flight_class']
    d['route_stops_class'] = d['route'] + '_' + d['stops'] + '_' + d['flight_class']
    d['dep_arr_time'] = d['departure_time'] + '_' + d['arrival_time']
    
    # 4. Duration & Speed Mechanics
    d['duration_minutes'] = np.round(d['duration'] * 60.0)
    d['duration_hours_int'] = np.floor(d['duration']).astype(int)
    d['duration_per_stop'] = d['duration'] / (d['stops_num'] + 1.0)
    d['effective_speed_mph'] = d['route_distance_miles'] / (d['duration'] + 0.1)
    d['is_Direct'] = (d['stops_num'] == 0).astype(int)
    d['is_long_layover'] = (d['duration'] > 6.0).astype(int)
    d['is_overnight_layover'] = (d['duration'] > 10.0).astype(int)
    
    # 5. Booking Lead-Time (`days_left`) Dynamic Pricing Features
    d['days_left_inv'] = 1.0 / (d['days_left'] + 1.0)
    d['days_left_log'] = np.log1p(d['days_left'])
    d['is_last_minute_1_3'] = (d['days_left'] <= 3).astype(int)
    d['is_urgent_4_7'] = ((d['days_left'] >= 4) & (d['days_left'] <= 7)).astype(int)
    d['is_advance_8_14'] = ((d['days_left'] >= 8) & (d['days_left'] <= 14)).astype(int)
    d['is_early_bird_21_plus'] = (d['days_left'] >= 21).astype(int)
    # Airline yield management bucket (0: 1-3d, 1: 4-7d, 2: 8-14d, 3: 15-21d, 4: 22-30d, 5: 31+d)
    d['booking_bucket'] = pd.cut(d['days_left'], bins=[0, 3, 7, 14, 21, 30, 100], labels=[0, 1, 2, 3, 4, 5]).astype(int)
    
    return d


def add_query_block_structure(train_df, test_df):
    comb = pd.concat([
        train_df.assign(_is_train=1, _orig_idx=np.arange(len(train_df))),
        test_df.assign(_is_train=0, _orig_idx=np.arange(len(test_df)))
    ], ignore_index=True).sort_values('id').reset_index(drop=True)
    
    q_change = (
        (comb['source_city'] != comb['source_city'].shift(1)) |
        (comb['destination_city'] != comb['destination_city'].shift(1)) |
        (comb['flight_class'] != comb['flight_class'].shift(1)) |
        (comb['days_left'] != comb['days_left'].shift(1))
    ).astype(int)
    comb['query_id'] = q_change.cumsum()
    
    b_stats = comb.groupby('query_id').agg(
        block_size=('id', 'count'),
        block_min_id=('id', 'min'),
        block_max_id=('id', 'max'),
        block_min_dur=('duration', 'min'),
        block_mean_dur=('duration', 'mean')
    ).reset_index()
    comb = comb.merge(b_stats, on='query_id', how='left')
    
    comb['pos_in_block'] = comb['id'] - comb['block_min_id']
    comb['rel_pos_in_block'] = comb['pos_in_block'] / np.maximum(comb['block_size'] - 1, 1)
    comb['is_first_in_block'] = (comb['id'] == comb['block_min_id']).astype(int)
    comb['is_last_in_block'] = (comb['id'] == comb['block_max_id']).astype(int)
    comb['zero_price_structural_flag'] = (
        (comb['is_last_in_block'] == 1) & 
        (comb['is_zero_price_candidate_flight'] == 1) & 
        (comb['stops_num'] == 1)
    ).astype(int)
    
    airline_q_stats = comb.groupby(['query_id', 'airline']).agg(
        airline_in_block_count=('id', 'count'),
        airline_in_block_min_id=('id', 'min')
    ).reset_index()
    comb = comb.merge(airline_q_stats, on=['query_id', 'airline'], how='left')
    comb['pos_in_airline_block'] = comb['id'] - comb['airline_in_block_min_id']
    
    # Sort back by `_orig_idx` so train and test row order are 100% preserved!
    tr_out = comb[comb['_is_train'] == 1].sort_values('_orig_idx').drop(columns=['_is_train', '_orig_idx']).reset_index(drop=True)
    te_out = comb[comb['_is_train'] == 0].sort_values('_orig_idx').drop(columns=['_is_train', '_orig_idx', 'price'], errors='ignore').reset_index(drop=True)
    return tr_out, te_out


def run_phase3_feature_engineering(train_df, test_df, preproc_state, n_splits=5, random_state=42):
    # 1. Deterministic Row-Level Domain Features
    tr = build_deterministic_features(train_df, preproc_state)
    te = build_deterministic_features(test_df, preproc_state)
    
    # 2. Search-Query Block Structural Features (using combined non-target features sorted by `id`)
    tr, te = add_query_block_structure(tr, te)
    
    # Refine the exact structural zero-price rule discovered in Train:
    # 23/23 (100%) of Southwest 1-stop SJC<->SMF flights at the end of a search block have price == 0
    for d in [tr, te]:
        d['is_wn_sjc_smf_last_in_block'] = (
            (d['is_last_in_block'] == 1) &
            (d['undirected_route'] == 'Sacramento_San Jose') &
            (d['airline'] == 'Southwest') &
            (d['stops_num'] == 1)
        ).astype(int)
        
    # 3. Fit Non-Target Group Aggregations (Duration & Frequency stats) on TRAIN ONLY and map to Train & Test
    # (Storing every lookup table in `fe_state` as requested so Train-derived stats transform Test cleanly)
    fe_state = {}
    
    # 3a. Route duration statistics (fitted on Train)
    route_dur_stats = tr.groupby('route')['duration'].agg(
        route_min_dur='min',
        route_mean_dur='mean',
        route_std_dur='std',
        route_max_dur='max'
    ).reset_index()
    route_dur_stats['route_std_dur'] = route_dur_stats['route_std_dur'].fillna(0.0)
    fe_state['route_dur_stats'] = route_dur_stats
    
    tr = tr.merge(route_dur_stats, on='route', how='left')
    te = te.merge(route_dur_stats, on='route', how='left')
    
    # 3b. Route + Airline + Stops duration statistics (fitted on Train)
    ras_dur_stats = tr.groupby(['route', 'airline', 'stops'])['duration'].agg(
        ras_min_dur='min',
        ras_mean_dur='mean'
    ).reset_index()
    fe_state['ras_dur_stats'] = ras_dur_stats
    
    tr = tr.merge(ras_dur_stats, on=['route', 'airline', 'stops'], how='left')
    te = te.merge(ras_dur_stats, on=['route', 'airline', 'stops'], how='left')
    
    for d in [tr, te]:
        d['ras_min_dur'] = d['ras_min_dur'].fillna(d['route_min_dur'])
        d['ras_mean_dur'] = d['ras_mean_dur'].fillna(d['route_mean_dur'])
        
        # Relative duration ratios and layover penalties
        d['dur_ratio_to_route_min'] = d['duration'] / (d['route_min_dur'] + 0.01)
        d['dur_diff_from_route_min'] = d['duration'] - d['route_min_dur']
        d['dur_zscore_in_route'] = (d['duration'] - d['route_mean_dur']) / (d['route_std_dur'] + 0.05)
        d['dur_diff_from_ras_mean'] = d['duration'] - d['ras_mean_dur']
        d['dur_diff_from_block_min'] = d['duration'] - d['block_min_dur']
        
    # 3c. Frequency Encoding fitted on Train (stored in fe_state)
    freq_cols = ['flight', 'route', 'airline', 'route_airline_class', 'flight_num']
    fe_state['freq_maps'] = {}
    for col in freq_cols:
        f_map = (tr[col].value_counts() / len(tr)).to_dict()
        fe_state['freq_maps'][col] = f_map
        tr[f'{col}_freq'] = tr[col].map(f_map).fillna(0.0)
        te[f'{col}_freq'] = te[col].map(f_map).fillna(0.0)
        
    # 4. Out-Of-Fold (OOF) Hierarchical Target Encoding & Block Neighbor Price Interpolation
    # Crucial: Only use `in-fold` training rows to compute target statistics for `val-fold` rows,
    # and use full `tr` (stored in `fe_state`) to transform `te`!
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fe_state['cv_splits'] = list(kf.split(tr))
    
    global_mean = float(tr['price'].mean())
    fe_state['global_price_mean'] = global_mean
    
    # Multi-granularity target encoding columns
    te_specs = [
        ('flight', 5.0),
        ('route_airline_class', 10.0),
        ('route_stops_class', 10.0),
        ('route_class', 10.0),
        ('airline_class', 10.0)
    ]
    
    # Initialize OOF columns in Train and Test
    for col, _ in te_specs:
        tr[f'{col}_te_mean'] = np.nan
        te[f'{col}_te_mean'] = np.nan
        if col in ['flight', 'route_airline_class', 'route_stops_class']:
            tr[f'{col}_te_std'] = np.nan
            tr[f'{col}_te_min'] = np.nan
            tr[f'{col}_te_max'] = np.nan
            te[f'{col}_te_std'] = np.nan
            te[f'{col}_te_min'] = np.nan
            te[f'{col}_te_max'] = np.nan
            
    # Also compute Out-Of-Fold Search-Query Block Price Anchors!
    # Within each `query_id` (and `query_id + airline`), what is the mean/min/max/nearest price of OTHER flights in the same search query?
    block_anchor_cols = [
        'qblock_oof_mean_price', 'qblock_oof_min_price', 'qblock_oof_max_price',
        'qblock_airline_oof_mean_price', 'qblock_prev_price', 'qblock_next_price', 'qblock_interp_price'
    ]
    for c in block_anchor_cols:
        tr[c] = np.nan
        te[c] = np.nan
        
    fe_state['te_maps'] = {}
    
    # Helper to compute block neighbor prices from a given reference set of labeled rows (`ref_df`) onto target rows (`target_df`)
    def compute_block_anchors(ref_df, target_df):
        # Exclude zero-price anomaly rows when computing normal block price anchors so 0 doesn't corrupt neighbors
        ref_valid = ref_df[ref_df['price'] > 0].copy()
        
        # 1. Query block summary stats
        q_stats = ref_valid.groupby('query_id')['price'].agg(
            q_mean='mean', q_min='min', q_max='max'
        )
        # 2. Query block + Airline summary stats
        qa_stats = ref_valid.groupby(['query_id', 'airline'])['price'].agg(
            qa_mean='mean'
        )
        
        res = target_df[['id', 'query_id', 'airline', 'route_airline_class', 'route_class']].copy()
        res = res.merge(q_stats, on='query_id', how='left')
        res = res.merge(qa_stats, on=['query_id', 'airline'], how='left')
        
        # 3. Exact nearest neighbor prices by `id` within the same `query_id`
        # Combine target `id`s and reference `(id, price)` sorted by `id` within `query_id`
        ref_lookup = ref_valid[['id', 'query_id', 'price']].rename(columns={'price': 'ref_price'})
        grid = pd.concat([
            res[['id', 'query_id']].assign(_is_target=1),
            ref_lookup.assign(_is_target=0)
        ], ignore_index=True).sort_values(['query_id', 'id', '_is_target']).reset_index(drop=True)
        
        grid['prev_p'] = grid.groupby('query_id')['ref_price'].ffill()
        grid['next_p'] = grid.groupby('query_id')['ref_price'].bfill()
        
        target_grid = grid[grid['_is_target'] == 1].drop_duplicates(subset=['id']).set_index('id')
        res['prev_p'] = res['id'].map(target_grid['prev_p'])
        res['next_p'] = res['id'].map(target_grid['next_p'])
        
        # Interpolated neighbor price (average of prev and next in block if both exist, else whichever exists)
        res['interp_p'] = res[['prev_p', 'next_p']].mean(axis=1)
        
        return res
    
    # Compute OOF features across the 5 folds for Train
    for tr_idx, val_idx in fe_state['cv_splits']:
        tr_fold = tr.iloc[tr_idx]
        tr_fold_nonzero = tr_fold[tr_fold['price'] > 0]
        val_fold = tr.iloc[val_idx]
        
        # Fallback hierarchy mean in this fold: route_class -> airline_class -> global_mean
        rc_means = tr_fold_nonzero.groupby('route_class')['price'].mean()
        ac_means = tr_fold_nonzero.groupby('airline_class')['price'].mean()
        
        for col, smooth in te_specs:
            stats = tr_fold_nonzero.groupby(col)['price'].agg(['count', 'mean', 'std', 'min', 'max'])
            # Smooth toward route_class mean where possible, else global_mean
            s_mean = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
            tr.iloc[val_idx, tr.columns.get_loc(f'{col}_te_mean')] = val_fold[col].map(s_mean)
            if f'{col}_te_std' in tr.columns:
                tr.iloc[val_idx, tr.columns.get_loc(f'{col}_te_std')] = val_fold[col].map(stats['std']).fillna(0.0)
                tr.iloc[val_idx, tr.columns.get_loc(f'{col}_te_min')] = val_fold[col].map(stats['min'])
                tr.iloc[val_idx, tr.columns.get_loc(f'{col}_te_max')] = val_fold[col].map(stats['max'])
                
        # Block anchors for val_fold using tr_fold
        anchors = compute_block_anchors(tr_fold, val_fold)
        tr.iloc[val_idx, tr.columns.get_loc('qblock_oof_mean_price')] = anchors['q_mean'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_oof_min_price')] = anchors['q_min'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_oof_max_price')] = anchors['q_max'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_airline_oof_mean_price')] = anchors['qa_mean'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_prev_price')] = anchors['prev_p'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_next_price')] = anchors['next_p'].values
        tr.iloc[val_idx, tr.columns.get_loc('qblock_interp_price')] = anchors['interp_p'].values
        
    # Now fit full-train Target Encodings & Block Anchors and apply to Test (storing in `fe_state`)
    tr_nonzero = tr[tr['price'] > 0]
    for col, smooth in te_specs:
        stats = tr_nonzero.groupby(col)['price'].agg(['count', 'mean', 'std', 'min', 'max'])
        s_mean = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
        fe_state['te_maps'][col] = {
            'mean': s_mean.to_dict(),
            'std': stats['std'].fillna(0.0).to_dict(),
            'min': stats['min'].to_dict(),
            'max': stats['max'].to_dict()
        }
        te[f'{col}_te_mean'] = te[col].map(fe_state['te_maps'][col]['mean'])
        if f'{col}_te_std' in te.columns:
            te[f'{col}_te_std'] = te[col].map(fe_state['te_maps'][col]['std']).fillna(0.0)
            te[f'{col}_te_min'] = te[col].map(fe_state['te_maps'][col]['min'])
            te[f'{col}_te_max'] = te[col].map(fe_state['te_maps'][col]['max'])
            
    test_anchors = compute_block_anchors(tr, te)
    te['qblock_oof_mean_price'] = test_anchors['q_mean'].values
    te['qblock_oof_min_price'] = test_anchors['q_min'].values
    te['qblock_oof_max_price'] = test_anchors['q_max'].values
    te['qblock_airline_oof_mean_price'] = test_anchors['qa_mean'].values
    te['qblock_prev_price'] = test_anchors['prev_p'].values
    te['qblock_next_price'] = test_anchors['next_p'].values
    te['qblock_interp_price'] = test_anchors['interp_p'].values
    
    # Fill any hierarchical NaNs cleanly (e.g. unseen flights in test fall back to route_airline_class_te_mean -> route_class_te_mean -> global_mean)
    for d in [tr, te]:
        d['route_class_te_mean'] = d['route_class_te_mean'].fillna(global_mean)
        d['airline_class_te_mean'] = d['airline_class_te_mean'].fillna(global_mean)
        d['route_stops_class_te_mean'] = d['route_stops_class_te_mean'].fillna(d['route_class_te_mean'])
        d['route_airline_class_te_mean'] = d['route_airline_class_te_mean'].fillna(d['route_stops_class_te_mean'])
        d['flight_te_mean'] = d['flight_te_mean'].fillna(d['route_airline_class_te_mean'])
        
        for col in ['flight', 'route_airline_class', 'route_stops_class']:
            d[f'{col}_te_std'] = d[f'{col}_te_std'].fillna(0.0)
            d[f'{col}_te_min'] = d[f'{col}_te_min'].fillna(d[f'{col}_te_mean'])
            d[f'{col}_te_max'] = d[f'{col}_te_max'].fillna(d[f'{col}_te_mean'])
            
        # Fill missing block neighbor prices with route_airline_class_te_mean
        d['qblock_oof_mean_price'] = d['qblock_oof_mean_price'].fillna(d['route_class_te_mean'])
        d['qblock_oof_min_price'] = d['qblock_oof_min_price'].fillna(d['route_class_te_mean'])
        d['qblock_oof_max_price'] = d['qblock_oof_max_price'].fillna(d['route_class_te_mean'])
        d['qblock_airline_oof_mean_price'] = d['qblock_airline_oof_mean_price'].fillna(d['route_airline_class_te_mean'])
        d['qblock_interp_price'] = d['qblock_interp_price'].fillna(d['qblock_oof_mean_price'])
        d['qblock_prev_price'] = d['qblock_prev_price'].fillna(d['qblock_interp_price'])
        d['qblock_next_price'] = d['qblock_next_price'].fillna(d['qblock_interp_price'])
        
        # Relative price position inside the query block range
        d['qblock_expected_price_by_pos'] = (
            d['qblock_oof_min_price'] + d['rel_pos_in_block'] * (d['qblock_oof_max_price'] - d['qblock_oof_min_price'])
        )
        
    # 5. Align Categorical dtypes identically between Train and Test for XGBoost native support
    cat_cols = [
        'airline', 'flight', 'flight_prefix', 'source_city', 'departure_time', 'stops',
        'arrival_time', 'destination_city', 'flight_class', 'route', 'undirected_route',
        'airline_class', 'route_class', 'route_airline_class', 'route_stops_class', 'dep_arr_time'
    ]
    fe_state['cat_cols'] = cat_cols
    fe_state['cat_categories'] = {}
    for col in cat_cols:
        all_cats = sorted(list(set(tr[col].astype(str)).union(set(te[col].astype(str)))))
        fe_state['cat_categories'][col] = all_cats
        tr[col] = pd.Categorical(tr[col].astype(str), categories=all_cats)
        te[col] = pd.Categorical(te[col].astype(str), categories=all_cats)
        
    return tr, te, fe_state


# ======================= MODIFIED TRAINING / SUBMISSION =======================
# Changes vs. original notebook:
#  1. Dropped 28 redundant / low-value engineered features (cyclical sin/cos, overlapping day-bucket flags, duplicate duration features, etc.)
#  2. Replaced the 6-config hyper-parameter search with ONE fixed config on log1p(price), higher learning rate, fewer trees.
#  3. Uses XGBoost if installed, otherwise falls back to sklearn HistGradientBoosting (same settings idea).
import os, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

DATA_DIR = '/mnt/user-data/uploads/' if os.path.exists('/mnt/user-data/uploads/train.csv') else '/kaggle/input/datasets/itzmahan/dataets101/'
OUT_DIR = '/mnt/user-data/outputs/' if os.path.isdir('/mnt/user-data/outputs') else '.'
train = pd.read_csv(DATA_DIR + 'train.csv'); test = pd.read_csv(DATA_DIR + 'test.csv'); sub = pd.read_csv(DATA_DIR + 'submission.csv')

train_clean, test_clean, preproc_state = run_phase2_preprocessing(train, test)
tr_fe, te_fe, fe_state = run_phase3_feature_engineering(train_clean, test_clean, preproc_state)

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_XGB = False

# high-cardinality categoricals (>255 levels) -> integer codes
for col in fe_state['cat_cols']:
    cats = fe_state['cat_categories'][col]
    if len(cats) > 255:
        tr_fe[col] = pd.Categorical(tr_fe[col].astype(str), categories=cats).codes
        te_fe[col] = pd.Categorical(te_fe[col].astype(str), categories=cats).codes

DROP = ['dep_sin','dep_cos','arr_sin','arr_cos','is_late_night_dep','is_peak_dep','is_red_eye_or_early',
        'is_last_minute_1_3','is_urgent_4_7','is_advance_8_14','is_early_bird_21_plus','days_left_inv','days_left_log',
        'duration_minutes','duration_hours_int','is_long_layover','is_overnight_layover','flight_num_tier',
        'is_regional_flight_num','src_is_north','dst_is_north','is_intra_region','is_first_in_block',
        'dur_zscore_in_route','dur_ratio_to_route_min','undirected_route','dep_arr_time','flight_prefix']
feature_cols = [c for c in tr_fe.columns if c not in {'id','price','query_id'} | set(DROP)]
y = tr_fe['price'].astype(float).to_numpy(); ly = np.log1p(np.maximum(y, 0))

def make_model():
    if USE_XGB:
        return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.07, max_depth=7,
                                min_child_weight=5, subsample=0.8, colsample_bytree=0.85, reg_lambda=8.0,
                                tree_method='hist', enable_categorical=True, random_state=42, n_jobs=-1)
    return HistGradientBoostingRegressor(learning_rate=0.08, max_iter=300, max_depth=7, min_samples_leaf=10,
                                         l2_regularization=8, early_stopping=False,
                                         categorical_features='from_dtype', random_state=42)

# 5-fold OOF check
oof = np.zeros(len(tr_fe))
for a, b in fe_state['cv_splits']:
    m = make_model(); m.fit(tr_fe.iloc[a][feature_cols], ly[a])
    oof[b] = np.expm1(m.predict(tr_fe.iloc[b][feature_cols]))
print(f"OOF R2: {r2_score(y, oof):.5f} | RMSE: {root_mean_squared_error(y, oof):.3f} | MAE: {mean_absolute_error(y, oof):.3f}")

# Final fit on all training rows + submission
final_model = make_model(); final_model.fit(tr_fe[feature_cols], ly)
pred = np.maximum(np.expm1(final_model.predict(te_fe[feature_cols])), 0)
pred_df = pd.DataFrame({'id': te_fe['id'].to_numpy(), 'price': pred})
submission = sub[['id']].merge(pred_df, on='id', how='left', validate='one_to_one')
assert not submission['price'].isna().any() and len(submission) == len(test)
submission.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False)
print('Saved submission.csv', submission.shape); print(submission.head()); print(submission['price'].describe())
