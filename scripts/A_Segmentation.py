import sys
import os
import csv
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "QtAgg"

print("Python version:", sys.version)
print("Matplotlib version:", matplotlib.__version__)
print("Matplotlib backend:", matplotlib.get_backend())

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

##############################################################################
# USER PARAMETERS
##############################################################################

SHOW_MODE = "both"
# "both"       => keep user lines + also draw smooth curve
# "only_curve" => remove user lines, only draw the smooth curve
# "only_lines" => keep user lines, do not draw the smooth curve

LINE_WIDTH    = 1     # thickness of polygon lines
MARKER_SIZE   = 3     # size of the hollow markers
REGION_ALPHA  = 0.3   # transparency of the filled polygon region

VIDEO_PATHS = [
    r"C:\Users\smara\miniconda3\envs\yolo_env\MyScripts\Segmentation-YOLO\Cam1_2024-05-01_12-44-08.mp4"
]
NUM_FRAMES_TO_SAMPLE = 26
SEGMENTATION_NAMES = ["right_thigh","right_shank","right_paw"]  # e.g. ["right_thigh","right_shank","right_paw"]

OUTPUT_JSON_PATH = "manual_segmentations.json"
OUTPUT_CSV_PATH  = "manual_segmentations.csv"  # We'll produce "manual_segmentations.xlsx"

SEGMENT_COLORS = {
    "right_thigh": "r",
    "right_shank": "g",
    "right_paw":   "b",
}

SAVE_EVERY_N_FRAMES = 2  # user sets how often we auto-save JSON + Excel

##############################################################################
# FRAME SAMPLING + LOADING
##############################################################################

def sample_frame_indices(num_frames, total_frames):
    """
    Uniformly sample 'num_frames' from [0, total_frames-1].
    If num_frames >= total_frames, returns all indices.
    """
    if num_frames <= 0:
        return []
    if num_frames >= total_frames:
        return list(range(total_frames))
    spacing = total_frames / float(num_frames)
    inds = [int(round(i*spacing)) for i in range(num_frames)]
    inds = sorted(set(min(ix, total_frames-1) for ix in inds))
    return inds

def load_video_frames(video_path, frame_indices):
    """Load frames from a video. Returns list of (frame_idx, frame_img)."""
    frames=[]
    cap=cv2.VideoCapture(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret,frame=cap.read()
        if ret:
            frames.append((idx,frame))
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")
    cap.release()
    return frames

##############################################################################
# MERGING OLD + NEW, PRINTING INFO
##############################################################################

def load_existing_json(json_path):
    if os.path.isfile(json_path):
        with open(json_path,'r') as f:
            return json.load(f)
    return []

def print_existing_info(all_data):
    """
    Print frames that already have segmentations for each video,
    ignoring those missing 'sample_frame_index'.
    """
    for vidinfo in all_data:
        vp=vidinfo["video_path"]
        segs=vidinfo["segmentation_data"]
        frames_with_segs=[]
        for sd in segs:
            if "sample_frame_index" in sd:
                frames_with_segs.append(sd["sample_frame_index"])
        frames_with_segs=sorted(set(frames_with_segs))
        print(f"Video: {vp}")
        print(f"  Already have segmentations for frames: {frames_with_segs}")

def merge_data(old_data, new_data):
    """
    Merge 'old_data' & 'new_data' by video_path,
    unify sample_frame_indices + segmentation_data.
    Overwrites same (frame, seg_name) if new.
    """
    merged_dict={}
    for vid in old_data:
        merged_dict[vid["video_path"]]=vid

    for vid in new_data:
        vp=vid["video_path"]
        if vp not in merged_dict:
            merged_dict[vp]=vid
        else:
            old_sf= set(merged_dict[vp]["sampled_frame_indices"])
            new_sf= set(vid["sampled_frame_indices"])
            merged_dict[vp]["sampled_frame_indices"]= sorted(list(old_sf.union(new_sf)))

            old_seg= merged_dict[vp]["segmentation_data"]
            seg_dict={}
            for sd in old_seg:
                fidx=sd.get("sample_frame_index",None)
                sname=sd.get("seg_name",None)
                poly= sd.get("polygon",[])
                pxls= sd.get("pixel_coords",[])
                if fidx is not None and sname is not None:
                    seg_dict[(fidx,sname)] = {
                        "polygon": poly,
                        "pixel_coords": pxls
                    }

            for nsd in vid["segmentation_data"]:
                nfidx= nsd.get("sample_frame_index",None)
                ns   = nsd.get("seg_name",None)
                poly2= nsd.get("polygon",[])
                px2  = nsd.get("pixel_coords",[])
                if nfidx is not None and ns is not None:
                    seg_dict[(nfidx,ns)] = {
                        "polygon": poly2,
                        "pixel_coords": px2
                    }

            merged_list=[]
            for(kf,ksn),val in seg_dict.items():
                merged_list.append({
                    "sample_frame_index":kf,
                    "seg_name":ksn,
                    "polygon": val["polygon"],
                    "pixel_coords": val["pixel_coords"]
                })
            merged_dict[vp]["segmentation_data"]= merged_list

    final=[]
    for vp, info in merged_dict.items():
        final.append(info)
    return final

def export_to_csv(all_data, csv_path):
    """
    Creates one Excel file with multiple sheets: one sheet per frame: 'frame_XXX'
    No main sheet. This can be slow if the data is large, so we do it less often.
    """
    import openpyxl
    from openpyxl import Workbook
    import os

    # gather data by frame
    frames_map = {}  # fidx -> list of [video_path, fidx, seg_name, x, y, type]
    for vid in all_data:
        vp= vid["video_path"]
        segs= vid["segmentation_data"]
        for sd in segs:
            fidx= sd.get("sample_frame_index",None)
            sname= sd.get("seg_name",None)
            poly= sd.get("polygon",[])
            pxls= sd.get("pixel_coords",[])
            if fidx is None or sname is None:
                continue
            # polygon points
            for (x,y) in poly:
                frames_map.setdefault(fidx,[]).append([vp,fidx,sname,x,y,"poly"])
            # pixel coords
            for(px,py) in pxls:
                frames_map.setdefault(fidx,[]).append([vp,fidx,sname,px,py,"pixel"])

    wb = Workbook()
    # remove default sheet so we only have one per frame
    default_sheet= wb.active
    wb.remove(default_sheet)

    # create a new sheet for each frame
    headers= ["video_path","sample_frame_index","seg_name","x","y","type"]
    for fidx, rowlist in frames_map.items():
        sheet_name= f"frame_{fidx}"
        wsf= wb.create_sheet(title=sheet_name)
        wsf.append(headers)
        for row in rowlist:
            wsf.append(row)

    # final .xlsx name
    xlsx_path = os.path.splitext(csv_path)[0]+".xlsx"
    wb.save(xlsx_path)
    print(f"**Wrote** => '{xlsx_path}' with multiple sheets (one per frame)")

##############################################################################
# ARROW KEYS + SCROLL => panning & zoom
##############################################################################

def handle_arrows_for_panning(event, ax):
    move_factor=0.05
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    w=xlim[1]-xlim[0]
    h=ylim[1]-ylim[0]
    if event.key=='left':
        ax.set_xlim(xlim[0]-w*move_factor,xlim[1]-w*move_factor)
    elif event.key=='right':
        ax.set_xlim(xlim[0]+w*move_factor,xlim[1]+w*move_factor)
    elif event.key=='up':
        ax.set_ylim(ylim[0]-h*move_factor,ylim[1]-h*move_factor)
    elif event.key=='down':
        ax.set_ylim(ylim[0]+h*move_factor,ylim[1]+h*move_factor)
    ax.figure.canvas.draw_idle()

def on_wheel_zoom(event, ax):
    if event.inaxes!=ax:
        return
    base_scale=1.1
    if event.button=='up':
        scale_factor=1/base_scale
    elif event.button=='down':
        scale_factor=base_scale
    else:
        return
    xdata,ydata=event.xdata,event.ydata
    if xdata is None or ydata is None:
        return
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    left   = xdata-(xdata - xlim[0])*scale_factor
    right  = xdata+(xlim[1] - xdata)*scale_factor
    bottom = ydata-(ydata - ylim[0])*scale_factor
    top    = ydata+(ylim[1] - ydata)*scale_factor
    ax.set_xlim(left,right)
    ax.set_ylim(bottom,top)
    ax.figure.canvas.draw_idle()

##############################################################################
# SMOOTH CLOSED CURVE
##############################################################################

def smooth_closed_curve(verts, num_points=200):
    if len(verts)<3:
        return verts
    x=np.array([v[0]for v in verts])
    y=np.array([v[1]for v in verts])
    x=np.append(x,x[0])
    y=np.append(y,y[0])
    tck,u=splprep([x,y],s=0,per=True)
    unew=np.linspace(0,1,num_points)
    out=splev(unew,tck)
    xs,ys=out[0],out[1]
    return list(zip(xs,ys))

##############################################################################
# FILL + PIXEL-FINDING
##############################################################################

def fill_polygon_transparent(ax, smoothed_pts, color, alpha):
    import matplotlib.patches as mpatches
    poly_patch= mpatches.Polygon(
        smoothed_pts, closed=True,
        facecolor=color, alpha=alpha,
        edgecolor='none'
    )
    ax.add_patch(poly_patch)

def find_pixels_in_polygon(smoothed_pts):
    path= Path(smoothed_pts, closed=True)
    xs=[p[0] for p in smoothed_pts]
    ys=[p[1] for p in smoothed_pts]
    min_x=int(min(xs)); max_x=int(max(xs))
    min_y=int(min(ys)); max_y=int(max(ys))
    pixel_coords=[]
    for px in range(min_x,max_x+1):
        for py in range(min_y,max_y+1):
            if path.contains_point( (px,py) ):
                pixel_coords.append((px,py))
    return pixel_coords

##############################################################################
# WAIT FOR SPACE => final shape remains visible
##############################################################################

def wait_for_enter(ax):
    """
    same code but using space => finalize seg & next frame
    """
    done=[False]
    def on_key(event):
        # use space => ' '
        if event.key==' ':
            done[0]=True
            ax.figure.canvas.stop_event_loop()

    cid= ax.figure.canvas.mpl_connect('key_press_event', on_key)
    ax.figure.canvas.start_event_loop(timeout=-1)
    ax.figure.canvas.mpl_disconnect(cid)

##############################################################################
# LEFT-CLICK-ONLY PolygonSelector => 'z' => undo => if last point near first => auto finalize => next seg
##############################################################################

from matplotlib.widgets import PolygonSelector

class LeftClickPolygonSelector(PolygonSelector):
    def _on_button_press(self, event):
        if event.button!=1:
            return
        super()._on_button_press(event)

    def _on_move(self, event):
        if self._active_handle is None and event.button!=1:
            return
        super()._on_move(event)

    def _on_button_release(self, event):
        super()._on_button_release(event)
        # if last point is close to first => finalize => next seg
        verts= list(self.verts)
        if len(verts)>2:
            first_pt= verts[0]
            last_pt = verts[-1]
            dist= np.hypot(first_pt[0]-last_pt[0], first_pt[1]-last_pt[1])
            if dist<5.0:
                self._selection_completed=True
                self.canvas.stop_event_loop()

def pick_polygon(ax, seg_name):
    """
    1) user draws => left-click => 'z'=undo => if last point near first => auto finalize
    2) smooth => fill => find pixels => show => wait => done => next seg.
    """
    raw_polygon=[]
    color= SEGMENT_COLORS.get(seg_name,'y')

    def onselect(verts):
        raw_polygon[:]=verts
        ax.figure.canvas.stop_event_loop()

    selector= LeftClickPolygonSelector(
        ax, onselect,
        useblit=False,
        props=dict(color=color, linewidth=LINE_WIDTH, alpha=0.8),
        handle_props=dict(marker='o', markeredgecolor=color,
                          markerfacecolor='none', markersize=MARKER_SIZE, alpha=0.8)
    )

    def on_key_undo(event):
        if event.key=='z':
            v=list(selector.verts)
            if v:
                v.pop()
                selector.verts=v
                selector._selection_completed=False
                selector._draw_polygon([])
                ax.figure.canvas.draw_idle()

        # if event.key==' '
        if event.key==' ':
            raw_polygon[:] = selector.verts
            ax.figure.canvas.stop_event_loop()

    cid_undo= ax.figure.canvas.mpl_connect('key_press_event', on_key_undo)

    ax.figure.canvas.start_event_loop(timeout=-1)
    ax.figure.canvas.mpl_disconnect(cid_undo)
    selector.disconnect_events()

    smoothed= smooth_closed_curve(raw_polygon,200)

    if SHOW_MODE=="only_lines":
        pass
    elif SHOW_MODE=="only_curve":
        for artist in selector.artists:
            artist.remove()
        draw_spline_on_axes(ax, smoothed, color)
    else: # both
        draw_spline_on_axes(ax, smoothed, color)

    fill_polygon_transparent(ax, smoothed, color, REGION_ALPHA)
    pixel_coords= find_pixels_in_polygon(smoothed)
    ax.figure.canvas.draw_idle()

    wait_for_enter(ax)  # user sees => press space => next seg
    return smoothed, pixel_coords

def draw_spline_on_axes(ax, smoothed_pts, color):
    if len(smoothed_pts)>1:
        sx=[p[0]for p in smoothed_pts]
        sy=[p[1]for p in smoothed_pts]
        sx.append(sx[0])
        sy.append(sy[0])
        ax.plot(sx, sy,
            color=color,
            linewidth=LINE_WIDTH,
            marker='o', markersize=MARKER_SIZE,
            markeredgecolor=color, markerfacecolor='none',
            alpha=0.8
        )

##############################################################################
# MAIN
##############################################################################

def main():
    # 1) Load existing data (if any)
    old_data= load_existing_json(OUTPUT_JSON_PATH)
    if old_data:
        print("Found existing manual_segmentations.json. Info:")
        print_existing_info(old_data)
    else:
        print("No existing data found; starting fresh.")

    updated_data= old_data[:]

    # 2) We'll accumulate changes in memory. Then only do a heavy save every N frames or final
    frames_since_save = 0  # count how many frames we've processed since last save

    # build a map => existing_frames_map[video_path] => set of frames
    existing_frames_map={}
    for vid in updated_data:
        vp=vid["video_path"]
        exist_frames=set()
        for sd in vid["segmentation_data"]:
            if "sample_frame_index" in sd:
                exist_frames.add(sd["sample_frame_index"])
        existing_frames_map[vp]= exist_frames

    for video_path in VIDEO_PATHS:
        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            continue

        cap=cv2.VideoCapture(video_path)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_indices= sample_frame_indices(NUM_FRAMES_TO_SAMPLE,total_frames)
        frames= load_video_frames(video_path,frame_indices)
        if not frames:
            print(f"No frames loaded for {video_path}")
            continue

        already_done_frames= existing_frames_map.get(video_path, set())

        for i,(f_idx,frame_img) in enumerate(frames):
            if f_idx in already_done_frames:
                print(f"\nFrame {f_idx} in {video_path} is already segmented => skipping.")
                continue

            real_num = f_idx+1
            print(f"\nVideo: {video_path}, Real frame {real_num}/{total_frames} => new seg needed.")

            fig, ax= plt.subplots(figsize=(8,6))
            ax.imshow(frame_img[..., ::-1])
            ax.set_title(
                f"Video: {os.path.basename(video_path)}\n"
                f"Frame {real_num}/{total_frames} (real idx {f_idx})\n"
                f"Space=finish, 'z'=undo, arrow=pan, scroll=zoom,\n"
                f"SHOW_MODE={SHOW_MODE}, ALPHA={REGION_ALPHA}"
            )
            plt.show(block=False)

            def on_key(event):
                if event.key not in ['z',' ']:
                    handle_arrows_for_panning(event, ax)
            cid_key=fig.canvas.mpl_connect('key_press_event', on_key)
            cid_scroll=fig.canvas.mpl_connect('scroll_event', lambda e:on_wheel_zoom(e,ax))

            for seg_name in SEGMENTATION_NAMES:
                ax.set_title(
                    f"Video:{os.path.basename(video_path)}, Frame {real_num}/{total_frames}\n"
                    f"Segment: {seg_name}, Space=finish, 'z'=undo\n"
                    f"SHOW_MODE={SHOW_MODE}, ALPHA={REGION_ALPHA}"
                )
                fig.canvas.draw_idle()

                smoothed_pts, pixel_coords= pick_polygon(ax, seg_name)

                # Merge & *not* immediately save
                single_seg_dict={
                    "video_path":video_path,
                    "sampled_frame_indices":[f_idx],
                    "segmentation_data":[
                        {
                            "sample_frame_index": f_idx,
                            "seg_name": seg_name,
                            "polygon": [(float(x),float(y)) for(x,y) in smoothed_pts],
                            "pixel_coords": [(int(px),int(py)) for(px,py) in pixel_coords]
                        }
                    ]
                }
                updated_data= merge_data(updated_data,[single_seg_dict])

                print(f"Segment '{seg_name}' on frame {f_idx} => done.\n"
                      f"Press SPACE or close window to proceed to next segment or next frame...")

            fig.canvas.mpl_disconnect(cid_key)
            fig.canvas.mpl_disconnect(cid_scroll)
            plt.close(fig)

            # Mark frame as done
            already_done_frames.add(f_idx)

            frames_since_save += 1
            # If we've processed N frames => auto-save
            if frames_since_save >= SAVE_EVERY_N_FRAMES:
                print(f"\nAuto-saving after {frames_since_save} frames...")
                with open(OUTPUT_JSON_PATH,'w') as ff:
                    json.dump(updated_data,ff,indent=2)
                export_to_csv(updated_data,OUTPUT_CSV_PATH)
                frames_since_save = 0  # reset

    # after finishing all frames
    # if frames_since_save != 0 => final save
    if frames_since_save > 0:
        print(f"\nFinal save after finishing all frames.")
        with open(OUTPUT_JSON_PATH,'w') as ff:
            json.dump(updated_data,ff,indent=2)
        export_to_csv(updated_data,OUTPUT_CSV_PATH)

    print("\nAll done! Data saved. Check your final results JSON + XLSX.")


if __name__=="__main__":
    main()
