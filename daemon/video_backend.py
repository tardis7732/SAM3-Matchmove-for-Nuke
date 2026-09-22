"""SAM3 video propagation from RAM frames, selecting identity only at reference."""
import numpy as np


def read_outputs(outputs, height, width):
    def array(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
    ids = [int(x) for x in array(outputs.get('out_obj_ids', [])).reshape(-1)]
    if not ids:
        return [], np.empty((0, height, width), bool), np.empty(0)
    masks = array(outputs['out_binary_masks'])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    scores = array(outputs['out_probs']).reshape(-1)
    if (masks.shape != (len(ids), height, width) or len(scores) != len(ids)
            or len(set(ids)) != len(ids) or not np.isfinite(scores).all()):
        raise ValueError('Invalid SAM3 video output')
    return ids, masks > 0, scores


def propagate(predictor, images, reference, prompt, confidence, object_index, cancelled):
    """Yield (index, mask, metadata); a missing selected ID yields black, never a substitute."""
    width, height = images[0].size
    session = None
    stream = None
    try:
        if cancelled():
            raise RuntimeError('Video analysis cancelled')
        session = predictor.handle_request(dict(type='start_session', resource_path=images,
            offload_video_to_cpu=True, offload_state_to_cpu=True))['session_id']
        response = predictor.handle_request(dict(type='add_prompt', session_id=session,
            frame_index=reference, text=prompt, output_prob_thresh=confidence))
        if response['frame_index'] != reference:
            raise ValueError('SAM3 returned the wrong reference frame')
        ids, masks, scores = read_outputs(response['outputs'], height, width)
        candidates = [i for i in range(len(ids)) if scores[i] >= confidence and masks[i].any()]
        candidates.sort(key=lambda i: (-int(masks[i].sum()), ids[i]))
        if not 0 <= object_index < len(candidates):
            raise ValueError('Object index %d is unavailable on the reference frame (%d targets). '
                             'Choose a reference where the target is visible.' % (object_index, len(candidates)))
        locked_id = ids[candidates[object_index]]

        def selected(outputs):
            ids, masks, scores = read_outputs(outputs, height, width)
            i = ids.index(locked_id) if locked_id in ids else None
            score = float(scores[i]) if i is not None else 0.0
            detected = i is not None and score >= confidence and bool(masks[i].any())
            mask = masks[i] if detected else np.zeros((height, width), bool)
            return mask, dict(object_id=locked_id, score=score, detected=detected, temporal_tracking=True)

        yield reference, *selected(response['outputs'])
        seen = {reference}
        stream = predictor.handle_stream_request(dict(type='propagate_in_video', session_id=session,
            propagation_direction='both', start_frame_index=reference, output_prob_thresh=confidence))
        for response in stream:
            if cancelled():
                raise RuntimeError('Video analysis cancelled')
            index = int(response['frame_index'])
            if not 0 <= index < len(images):
                raise ValueError('SAM3 returned an out-of-range frame')
            if index != reference:
                yield index, *selected(response['outputs'])
            seen.add(index)
        if len(seen) != len(images):
            raise RuntimeError('SAM3 video propagation did not complete the frame range')
    finally:
        try:
            if stream is not None and hasattr(stream, 'close'):
                stream.close()
        finally:
            if session is not None:
                predictor.handle_request(dict(type='close_session', session_id=session))


def track(checkpoint, images, reference, prompt, confidence, object_index, cancelled):
    import gc
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    predictor = None
    try:
        # The official predictor owns its autocast context; build and close on this thread.
        with torch.inference_mode():
            predictor = build_sam3_video_predictor(checkpoint_path=checkpoint, gpus_to_use=[0],
                                                   compile=False, async_loading_frames=False)
            yield from propagate(predictor, images, reference, prompt, confidence, object_index, cancelled)
    finally:
        if predictor is not None:
            predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
