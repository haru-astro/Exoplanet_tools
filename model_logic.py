import numpy as np
import math
import pytransit

def calc_transit_params(k, ln_a, T14, p_orbit):
    """
    接触時間 T14 から幾何学的パラメータ (a, i) を算出する。
    """
    a = math.exp(ln_a)
    # トランジット幾何学からインパクトパラメータ b を導出
    # x は接触時間に対応する位相距離
    x = a * math.sin(math.pi * T14 / p_orbit)
    
    if (1 + k) <= x:
        return None
    
    # 以下の式を用いて b を計算
    # b^2 = ((1+k)^2 - x^2) / (1 - (x/a)^2)
    b2 = (-(x**2) + (1 + k)**2) / (-(x**2 / a**2) + 1)
    
    if b2 < 0:
        return None
    
    b = math.sqrt(b2)
    
    # 物理的境界条件のチェック
    if b + k >= 1 or b / a >= 1:
        return None
    
    # 軌道傾斜角 i を算出
    inc = math.acos(b / a)
    return a, inc

def unpack_params(params, n_obs, n_events):
    """
    1次元のパラメータ配列を物理変数に展開する。
    t0 はイベント数 (n_events)、jitter はデータ数 (n_obs) に基づく。
    """
    # 共通物理パラメータ (7個)
    k, ln_a, T14 = params[0:3]
    ldc = {"i": params[3:5], "r": params[5:7]}
    
    # イベントごとの中心時刻 (n_events個)
    t0_list = params[7 : 7 + n_events]
    
    # データセットごとのジッター (n_obs個)
    jitter_list = params[7 + n_events : 7 + n_events + n_obs]
    
    return k, ln_a, T14, ldc, t0_list, jitter_list

def log_prior(params, n_obs, n_events, ldc_ref, init_vals):
    """
    事前分布の計算。
    """
    k, ln_a, T14, ldc, t0_list, jitter_list = unpack_params(params, n_obs, n_events)
    lp = 0
    
    # 周辺減光係数のガウス事前分布 (LDTKの計算結果に基づく)
    for b in ['i', 'r']:
        lp += -0.5 * np.sum(((ldc[b] - ldc_ref[b]["mean"]) / ldc_ref[b]["std"])**2)
    
    # k と T14 のガウス事前分布 (main.py で定義された誤差を使用)
    lp += -0.5 * (k - init_vals['k'])**2 / (init_vals['k_err'])**2
    lp += -0.5 * (T14 - init_vals['T14'])**2 / (init_vals['T14_err'])**2
    
    # ジッターの非負制約
    if any(j < 0 for j in jitter_list):
        return -np.inf
    
    return lp

def log_likelihood(params, n_obs, n_events, event_map, all_time, all_flux, all_flux_err, all_bands, best_degrees, p_orbit):
    """
    尤度関数の計算。event_map を使用して共通の t0 を各観測に割り当てる。
    """
    k, ln_a, T14, ldc, t0_list, jitter_list = unpack_params(params, n_obs, n_events)
    
    # 幾何学パラメータの更新
    geom = calc_transit_params(k, ln_a, T14, p_orbit)
    if geom is None:
        return -np.inf
    a, inc = geom
    
    ll = 0
    tm = pytransit.QuadraticModel()
    
    for i in range(n_obs):
        # event_map[i] により、その観測データに対応する共通 t0 を取得
        idx_t0 = event_map[i]
        t0 = t0_list[idx_t0]
        
        # モデルフラックスの計算
        tm.set_data(all_time[i])
        mtr = tm.evaluate_ps(k, ldc[all_bands[i]], t0, p_orbit, a, inc, 0, 0)
        
        # 多項式によるデトレンド
        c = np.polyfit(all_time[i] - t0, all_flux[i] / mtr, best_degrees[i])
        m_full = mtr * np.polyval(c, all_time[i] - t0)
        
        # カイ二乗尤度の累積 (ジッター項を含む)
        sig2 = all_flux_err[i]**2 + jitter_list[i]**2
        ll += -0.5 * np.sum((all_flux[i] - m_full)**2 / sig2 + np.log(2 * np.pi * sig2))
        
    return ll

def log_posterior(params, n_obs, n_events, event_map, all_time, all_flux, all_flux_err, all_bands, ldc_ref, init_vals, best_degrees, p_orbit):
    """
    事後分布の計算。
    """
    lp = log_prior(params, n_obs, n_events, ldc_ref, init_vals)
    if not np.isfinite(lp):
        return -np.inf
        
    ll = log_likelihood(params, n_obs, n_events, event_map, all_time, all_flux, all_flux_err, all_bands, best_degrees, p_orbit)
    
    return lp + ll