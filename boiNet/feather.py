import torch

def create_fg_masks(bboxes, image_height=218, image_width=178):
    """
    Create foreground masks for a batch of images given their bounding boxes, vectorized version.
    """
    batch_size = bboxes.size(0)
    # Create coordinate grids
    x_coords = torch.arange(image_width).repeat(image_height, 1).unsqueeze(0).repeat(batch_size, 1, 1)
    y_coords = torch.arange(image_height).repeat(image_width, 1).t().unsqueeze(0).repeat(batch_size, 1, 1)

    # Get bbox coordinates and expand dimensions for broadcasting
    lefts = bboxes[:, 0].unsqueeze(1).unsqueeze(2)
    tops = bboxes[:, 1].unsqueeze(1).unsqueeze(2)
    rights = bboxes[:, 2].unsqueeze(1).unsqueeze(2)
    bottoms = bboxes[:, 3].unsqueeze(1).unsqueeze(2)

    # Create masks using logical operations
    masks = (x_coords >= lefts) & (x_coords < rights) & (y_coords >= tops) & (y_coords < bottoms)
    masks = masks.float().unsqueeze(1)  # Convert from bool to float and add channel dimension

    return masks >= 1


def linear_feather_mask(size, feather_size):
    """Create a rectangular feather mask which fades in from all sides. FG = 1, BG = 0."""
    
    # Vertical and horizontal gradients
    h, w = size
    u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
    v = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
    
    # Combine them to get a gradient that fades in from all sides
    mask = torch.min(u, torch.min(1-u, torch.min(v, 1-v)))
    mask = torch.clamp(mask * max(w, h) / feather_size, 0, 1)
    
    return mask


def stitch(fg, bg, bboxes, feather_size=3):
    """Given batched fg, bg, and bboxes coordinates, stitch fg and bg
    together with feathering.
    """
    N, _, H, W = bg.shape
    
    mask_fg = linear_feather_mask(fg.shape[2:], feather_size)
    mask_fg = mask_fg.expand((N, 1, fg.shape[2], fg.shape[3]))
    
    binary_fg_mask = create_fg_masks(bboxes, H, W)
    mask_bg = torch.ones((N, 1, H, W))
    mask_bg[binary_fg_mask] = (1 - mask_fg).flatten()

    rec = bg * mask_bg
    rec[binary_fg_mask.expand_as(bg)] += (fg * mask_fg).flatten()
    
    return rec


