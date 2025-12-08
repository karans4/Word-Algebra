import struct
import base64

#Save state ID for next
def generate_state_id(start, end, depth, words_list, vocab_map):
    """
    Serializes state into a compact, URL-safe Base64 string.
    Format: [Version:1B][Depth:1B][StartID:4B][EndID:4B][ListLen:2B][ListIDs:4B*N]
    """
    VERSION = 1
    
    # 1. Cleaning and Lookup (O(1))
    # Using .get(x, 0) defaults to 0 if word not found. 
    # Ideally, ensure words are valid before calling this.
    s_idx = vocab_map.get(start.lower().strip(), 0)
    e_idx = vocab_map.get(end.lower().strip(), 0)
    
    # Convert list of words to list of integers
    # We strip/lower only once here
    w_idxs = [vocab_map.get(w.lower().strip(), 0) for w in words_list]

    # 2. Binary Packing (The "C" way)
    # > = Big Endian (Standard for network/portable data)
    # B = Unsigned Char (1 byte)
    # I = Unsigned Int (4 bytes)
    # H = Unsigned Short (2 bytes)
    
    # Pack the Header: Version, Depth, Start Index, End Index, Count of Words
    header = struct.pack('>BBIIH', VERSION, int(depth), s_idx, e_idx, len(w_idxs))
    
    # Pack the Body: The list of neighbor indexes
    # f'>{len(w_idxs)}I' creates a format string like '>5I' for 5 integers
    body = struct.pack(f'>{len(w_idxs)}I', *w_idxs)

    # 3. Encode to Base64 (URL Safe)
    # We decode to 'ascii' at the end to return a standard python string, not bytes
    return base64.urlsafe_b64encode(header + body).decode('ascii')
    
    
def decode_state_id(state_id, index_to_word_map):
    try:
        # 1. Decode Base64 to Bytes
        raw_bytes = base64.urlsafe_b64decode(state_id)
        
        # 2. Unpack Header (First 12 bytes: 1+1+4+4+2)
        version, depth, s_idx, e_idx, list_len = struct.unpack('>BBIIH', raw_bytes[:12])
        
        # Version Check
        if version != 1: raise ValueError("Unsupported Version")

        # 3. Unpack Body
        w_idxs = struct.unpack(f'>{list_len}I', raw_bytes[12:])
        
        # 4. Reconstruct Words
        return {
            "start": index_to_word_map.get(s_idx, "???"),
            "end": index_to_word_map.get(e_idx, "???"),
            "depth": depth,
            "path": [index_to_word_map.get(i, "???") for i in w_idxs]
        }
    except Exception as e:
        print(f"Invalid State ID: {e}")
        return None