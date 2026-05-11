import cv2, numpy as np

def extract_lgh_sequence(img, w_width=16, cells_y=4, cells_x=4, bins=8):
    """Imagen normalizada (texto negro / fondo blanco) -> lista de vectores LGH."""
    img_inv = cv2.bitwise_not(img)
    h, w    = img_inv.shape
    if w < w_width:
        return []

    angle_step  = 2 * np.pi / bins
    bin_centers = np.linspace(0, 2 * np.pi, bins, endpoint=False)
    smoothed    = cv2.GaussianBlur(img_inv, (5, 5), 0).astype(np.float64)
    gx = np.zeros_like(smoothed); gy = np.zeros_like(smoothed)
    gx[:, 1:-1] = smoothed[:, 2:] - smoothed[:, :-2]
    gy[1:-1, :]  = smoothed[2:, :] - smoothed[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx); ang[ang < 0] += 2 * np.pi

    word_features = []
    for x_start in range(w - w_width + 1):
        win_img = img_inv[:, x_start:x_start + w_width]
        win_mag = mag[:,   x_start:x_start + w_width]
        win_ang = ang[:,   x_start:x_start + w_width]
        y_idx   = np.where(np.any(win_img > 0, axis=1))[0]
        fv      = np.zeros(cells_y * cells_x * bins)

        if len(y_idx) > 0:
            y_min, y_max = y_idx[0], y_idx[-1]
            cell_h = (y_max - y_min + 1) / cells_y
            cell_w = w_width / cells_x
            vi = 0
            for r in range(cells_y):
                for c in range(cells_x):
                    rs = y_min + int(r * cell_h)
                    re = y_min + int((r+1)*cell_h) if r < cells_y-1 else y_max+1
                    cs = int(c * cell_w)
                    ce = int((c+1)*cell_w) if c < cells_x-1 else w_width
                    cm = win_mag[rs:re, cs:ce].flatten()
                    ca = win_ang[rs:re, cs:ce].flatten()
                    hist = np.zeros(bins)
                    for i in range(len(cm)):
                        m, theta = cm[i], ca[i]
                        if m == 0:
                            continue
                        d = np.abs(theta - bin_centers)
                        d = np.minimum(d, 2*np.pi - d)
                        n2 = np.argsort(d)[:2]; alpha = d[n2[0]]
                        hist[n2[0]] += m * (1.0 - alpha / angle_step)
                        hist[n2[1]] += m * (alpha / angle_step)
                    fv[vi:vi+bins] = hist; vi += bins

        s = np.sum(fv)
        word_features.append(fv / s if s > 1e-7 else np.zeros_like(fv))
    return word_features
