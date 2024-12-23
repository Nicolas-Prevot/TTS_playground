import argparse
import os
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, dash_table
import plotly.graph_objs as go
import numpy as np
import wave
import struct
from pathlib import Path
import base64
from loguru import logger

from .utils_audio import run_ffmpeg_command, normalize_audio


def read_audio_file(file_path):
    """Read a WAV file and return time array, data array, and format info."""
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if samp_width == 1:
        fmt_char = 'b'
    elif samp_width == 2:
        fmt_char = 'h'
    elif samp_width == 4:
        fmt_char = 'i'
    else:
        raise ValueError("Unsupported sample width")

    total_samples = n_frames * n_channels
    fmt = "<" + fmt_char * total_samples
    data = struct.unpack(fmt, raw_data)

    # For waveform visualization, just use the first channel
    if n_channels > 1:
        data = data[0::n_channels]

    data = np.array(data)
    duration = n_frames / float(framerate)
    times = np.linspace(0, duration, num=len(data))

    return times, data, framerate, n_channels, samp_width


def create_segment_file(segment, index, temp_dir, input_file, framerate, n_channels, ffmpeg_path):
    """Create a temporary WAV file for a single segment or silence."""
    out_file = os.path.join(temp_dir, f"segment_{index}.wav")
    #if out_file.exists():
    #    out_file.unlink()

    if segment['type'] == 'segment':
        start = segment['start']
        end = segment['end']
        duration = end - start
        if duration <= 0:
            raise ValueError("Invalid segment duration.")
        cmd = [
            ffmpeg_path, '-y', '-i', input_file,
            '-ss', str(start), '-t', str(duration),
            '-ar', str(framerate),      # match original sample rate
            '-ac', str(n_channels),     # match original channel count
            '-c:a', 'pcm_s16le',        # consistent PCM format
            # '-c', 'copy',
            str(out_file)
        ]
    elif segment['type'] == 'silence':
        duration = segment['duration']
        channel_layout = "stereo" if n_channels == 2 else "mono"
        cmd = [
            ffmpeg_path, '-y', '-f', 'lavfi',
            '-i', f"anullsrc=r={framerate}:cl={channel_layout}",
            '-t', str(duration),
            '-ar', str(framerate),   # ensure sample rate matches original
            '-ac', str(n_channels),  # ensure channel count matches original
            '-c:a', 'pcm_s16le',     # consistent PCM format
            str(out_file)
        ]
    run_ffmpeg_command(cmd)
    return out_file


def generate_preview(segments, framerate, n_channels, temp_dir, input_file, ffmpeg_path):
    """Generate a preview WAV file by concatenating all segments."""
    print(segments)
    preview_file = os.path.join(temp_dir, "preview_temp.wav")
    final_preview = os.path.join(temp_dir, "final_preview.wav")
    # if os.path.exists(preview_file):
    #     preview_file.unlink()
    # if os.path.exists(final_preview):
    #     final_preview.unlink()

    # Clean up old temp files
    #for f in temp_dir.glob("segment_*.wav"):
    #    f.unlink()

    segment_files = [create_segment_file(seg, i, temp_dir, input_file, framerate, n_channels, ffmpeg_path) for i, seg in enumerate(segments)]

    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, 'w') as f:
        for segf in segment_files:
            f.write(f"file '{Path(segf).resolve()}'\n")

    # First, concatenate segments
    cmd_concat = [
        ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_file),
        '-c', 'copy', str(preview_file)
    ]
    run_ffmpeg_command(cmd_concat)

    # Re-encode the concatenated file to a standard PCM WAV
    cmd_encode = [
        ffmpeg_path, '-y', '-i', str(preview_file),
        '-c:a', 'pcm_s16le',
        str(final_preview)
    ]
    run_ffmpeg_command(cmd_encode)

    return final_preview


def file_to_data_uri(file_path):
    """Convert a file to a base64 data URI for immediate use in src attributes."""
    with open(file_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode('ascii')
    return f"data:audio/wav;base64,{encoded}"



def create_dash_app(assets_dir, temp_dir, input_file, ffmpeg_path):

    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    times, data, framerate, n_channels, samp_width = read_audio_file(input_file)

    # Create initial waveform figure
    fig = go.Figure(data=go.Scatter(x=times, y=data, mode='lines', line=dict(width=1)))
    fig.update_layout(
        title="Audio Waveform (Use Box/Lasso Select Tool)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        dragmode='select',
        selectdirection='h'
    )

    app = dash.Dash(__name__)

    columns = [
        {"name": "Type", "id": "type", "presentation": "dropdown", "editable": False},
        {"name": "Start (s)", "id": "start", "type": "numeric", "editable": True},
        {"name": "End (s)", "id": "end", "type": "numeric", "editable": True},
        {"name": "Duration (s)", "id": "duration", "type": "numeric", "editable": False}
    ]

    initial_data = []

    app.layout = html.Div([
        html.H1("Advanced Audio Editor"),
        html.Div([
            dcc.Graph(
                id='waveform-graph',
                figure=fig,
                config={'modeBarButtonsToAdd': ['select2d', 'lasso2d'], 'displayModeBar': True},
                style={'border': '1px solid #ccc'}
            ),
            html.Button("Add Selected Segment", id='add-selected-segment', n_clicks=0, style={'margin-top':'10px'}),
        ], style={'width': '80%', 'margin': 'auto'}),

        html.Div([
            html.Label("Add Silence:"),
            dcc.Input(id='silence-input', type='number', min=0, step=0.1, value=0, style={'margin-left':'10px', 'width':'100px'}),
            html.Button("Add Silence", id='add-silence-button', n_clicks=0, style={'margin-left':'10px'}),
        ], style={'width': '80%', 'margin': 'auto', 'margin-top':'20px'}),

        html.Div([
            dash_table.DataTable(
                id='segments-table',
                columns=columns,
                data=initial_data,
                editable=True,
                row_deletable=False,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                dropdown={
                    'type': {
                        'options': [
                            {'label': 'segment', 'value': 'segment'},
                            {'label': 'silence', 'value': 'silence'}
                        ]
                    }
                }
            )
        ], style={'width': '80%', 'margin': 'auto', 'margin-top': '20px'}),

        html.Div([
            html.Button("Move Up", id='move-up', n_clicks=0),
            html.Button("Move Down", id='move-down', n_clicks=0, style={'margin-left':'10px'}),
            html.Button("Delete Selected", id='delete-selected', n_clicks=0, style={'margin-left':'10px'}),
            html.Button("Apply Edits", id='apply-edits', n_clicks=0, style={'margin-left':'10px', 'background-color':'#efefef'})
        ], style={'width': '80%', 'margin': 'auto', 'margin-top': '10px'}),

        html.Div([
            html.Button("Generate Preview", id='generate-preview', n_clicks=0),
            html.Button("Save Final Output", id='save-output', n_clicks=0, style={'margin-left':'10px'}),
            html.Div(id='action-status', style={'margin-top': '10px'}),
            html.Audio(id='audio-player', controls=True, src="", style={'margin-top':'20px', 'width':'80%'})
        ], style={'width': '80%', 'margin': 'auto', 'margin-top': '20px'}),

        dcc.Store(id='segments-data', data=initial_data),
        dcc.Store(id='selected-row', data=None),
        dcc.Store(id='graph-selection', data=None),
    ])

    @app.callback(
        Output('graph-selection', 'data'),
        Input('waveform-graph', 'selectedData')
    )
    def store_graph_selection(selectedData):
        """Store the currently selected time range from the graph."""
        if selectedData:
            if 'range' in selectedData and 'x' in selectedData['range']:
                x_range = selectedData['range']['x']
                if x_range and len(x_range) == 2:
                    start, end = sorted(x_range)
                    if end > start:
                        return {'start': start, 'end': end}

            elif 'points' in selectedData and len(selectedData['points']) > 0:
                xs = [p['x'] for p in selectedData['points'] if 'x' in p]
                if xs:
                    start, end = min(xs), max(xs)
                    if end > start:
                        return {'start': start, 'end': end}
        return None

    @app.callback(
        Output('segments-data', 'data'),
        Input('add-selected-segment', 'n_clicks'),
        Input('add-silence-button', 'n_clicks'),
        Input('move-up', 'n_clicks'),
        Input('move-down', 'n_clicks'),
        Input('delete-selected', 'n_clicks'),
        Input('apply-edits', 'n_clicks'),
        State('graph-selection', 'data'),
        State('silence-input', 'value'),
        State('segments-data', 'data'),
        State('selected-row', 'data'),
        State('segments-table', 'data'),
        prevent_initial_call=True
    )
    def modify_segments_data(add_seg_click, add_sil_click, move_up_click, move_down_click, delete_click,
                            apply_edits_click, selection, silence_val, segments, selected_row, table_data):
        """Single callback that updates segments-data from buttons or from applied edits."""
        ctx = callback_context
        if not ctx.triggered:
            return segments

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        updated_segments = segments.copy()

        if trigger_id == 'apply-edits':
            # User clicked "Apply Edits", update segments from table data
            temp_segments = []
            for seg in table_data:
                seg_copy = seg.copy()
                if seg_copy['type'] == 'segment' and seg_copy['start'] is not None and seg_copy['end'] is not None:
                    seg_copy['duration'] = round(seg_copy['end'] - seg_copy['start'], 2)
                temp_segments.append(seg_copy)
            updated_segments = temp_segments
        else:
            # Handle button actions
            if trigger_id == 'add-selected-segment' and selection:
                updated_segments.append({
                    'type': 'segment',
                    'start': round(selection['start'], 2),
                    'end': round(selection['end'], 2),
                    'duration': round(selection['end'] - selection['start'], 2)
                })

            elif trigger_id == 'add-silence-button' and silence_val is not None and silence_val > 0:
                updated_segments.append({
                    'type': 'silence',
                    'start': None,
                    'end': None,
                    'duration': round(silence_val, 2)
                })

            elif trigger_id == 'move-up' and updated_segments and selected_row is not None and 0 < selected_row < len(updated_segments):
                updated_segments[selected_row], updated_segments[selected_row - 1] = updated_segments[selected_row - 1], updated_segments[selected_row]

            elif trigger_id == 'move-down' and updated_segments and selected_row is not None and 0 <= selected_row < (len(updated_segments) - 1):
                updated_segments[selected_row], updated_segments[selected_row + 1] = updated_segments[selected_row + 1], updated_segments[selected_row]

            elif trigger_id == 'delete-selected' and updated_segments and selected_row is not None and 0 <= selected_row < len(updated_segments):
                updated_segments.pop(selected_row)

        return updated_segments

    @app.callback(
        Output('segments-table', 'data'),
        Input('segments-data', 'data')
    )
    def update_table_from_data(segments):
        """Keep the table synchronized with segments-data."""
        return segments

    @app.callback(
        Output('selected-row', 'data'),
        Input('segments-table', 'active_cell'),
        prevent_initial_call=True
    )
    def store_selected_row(active_cell):
        """Store the currently selected row index in the table."""
        if active_cell:
            return active_cell['row']
        return None

    @app.callback(
        Output('action-status', 'children'),
        Output('audio-player', 'src'),
        Input('generate-preview', 'n_clicks'),
        Input('save-output', 'n_clicks'),
        State('segments-data', 'data'),
        prevent_initial_call=True
    )
    def handle_preview_save(preview_click, save_click, segments):
        """Handle generating a preview or saving the final output."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'generate-preview':
            if not segments:
                return "No segments to preview.", no_update
            
            final_preview = generate_preview(segments, framerate, n_channels, temp_dir, input_file, ffmpeg_path)
            data_uri = file_to_data_uri(final_preview)
            return "Preview generated.", data_uri

        elif button_id == 'save-output':
            if not segments:
                return "No segments to save.", no_update
            
            final_preview = generate_preview(segments, framerate, n_channels, temp_dir, input_file, ffmpeg_path)
            final_file = os.path.join(assets_dir, "final_output.wav")
            normalize_audio(final_preview, final_file, target_db=-20)

            data_uri = file_to_data_uri(final_preview)
            return f"Final output saved as {os.path.basename(final_file)}.", data_uri

    @app.callback(
        Output('waveform-graph', 'figure'),
        Input('selected-row', 'data'),
        State('segments-data', 'data'),
        prevent_initial_call=True
    )
    def highlight_selected_segment(selected_row, segments):
        """Highlight the selected segment on the waveform graph."""
        new_fig = go.Figure(data=go.Scatter(x=times, y=data, mode='lines', line=dict(width=1)))
        new_fig.update_layout(
            title="Audio Waveform (Use Box/Lasso Select Tool)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300,
            dragmode='select',
            selectdirection='h'
        )

        if selected_row is not None and 0 <= selected_row < len(segments):
            seg = segments[selected_row]
            if seg['type'] == 'segment' and seg['start'] is not None and seg['end'] is not None:
                new_fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=seg['start'],
                    x1=seg['end'],
                    y0=0,
                    y1=1,
                    fillcolor="yellow",
                    opacity=0.3,
                    line_width=0
                )
        return new_fig
    
    return app


def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Advanced Audio Editor Web Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help="Path to the input audio file (e.g., .wav, .m4a)."
    )

    parser.add_argument(
        '--ffmpeg-path',
        type=str,
        default='ffmpeg',
        help="Path to the ffmpeg executable. Defaults to 'ffmpeg' if in PATH."
    )

    parser.add_argument(
        '--assets-dir',
        type=str,
        default="data/raw_audio",
        help="Directory to store asset files."
    )

    parser.add_argument(
        '--temp-dir',
        type=str,
        default="data/temp",
        help="Directory to store temporary files."
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Host address to run the Dash server."
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help="Port number to run the Dash server."
    )

    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help="Run the Dash server in debug mode."
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    logger.debug(f"Received arguments: {args}")

    app = create_dash_app(assets_dir=args.assets_dir,
                          temp_dir=args.temp_dir,
                          input_file=args.input_file,
                          ffmpeg_path=args.ffmpeg_path,
                          )
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
