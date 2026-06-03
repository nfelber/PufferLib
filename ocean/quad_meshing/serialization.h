#include <stdint.h>
#include <string.h>

// Little-endian
typedef struct {
    unsigned char *data;
    size_t pos;
} SerialBuffer;

static inline void serialize_u8(SerialBuffer *b, uint8_t v)
{
    b->data[b->pos++] = v;
}

static inline uint8_t deserialize_u8(SerialBuffer *b)
{
    return b->data[b->pos++];
}

static inline void serialize_u16(SerialBuffer *b, uint16_t v)
{
    b->data[b->pos++] = (unsigned char)(v & 0xFF);
    b->data[b->pos++] = (unsigned char)((v >> 8) & 0xFF);
}

static inline uint16_t deserialize_u16(SerialBuffer *b)
{
    uint16_t v =
        ((uint16_t)b->data[b->pos]) |
        ((uint16_t)b->data[b->pos + 1] << 8);

    b->pos += 2;

    return v;
}

static inline void serialize_float(SerialBuffer *b, float v)
{
    uint32_t bits;

    memcpy(&bits, &v, sizeof(bits));

    b->data[b->pos++] = bits & 0xFF;
    b->data[b->pos++] = (bits >> 8) & 0xFF;
    b->data[b->pos++] = (bits >> 16) & 0xFF;
    b->data[b->pos++] = (bits >> 24) & 0xFF;
}

static inline float deserialize_float(SerialBuffer *b)
{
    uint32_t bits =
        ((uint32_t)b->data[b->pos]) |
        ((uint32_t)b->data[b->pos + 1] << 8) |
        ((uint32_t)b->data[b->pos + 2] << 16) |
        ((uint32_t)b->data[b->pos + 3] << 24);

    b->pos += 4;

    float v;
    memcpy(&v, &bits, sizeof(bits));
    return v;
}

