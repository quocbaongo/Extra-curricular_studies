library(GenomicRanges)

# A way to compress a vector
rl = Rle(c(1,1,1,1,1,1,2,2,2,2,2,4,4,2))
rl 

runLength(rl)
runValue(rl)

# To get original input vector type:
as.numeric(rl)

ir = IRanges(start = c(2,8), width=4)

# mean of all the elements "rl" with range defined in "ir"
aggregate(rl, ir, FUN=mean)

# Output of "aggregate(rl, ir, FUN=mean)" is equivalent to
vec = as.numeric(rl)
mean(vec[2:5])
mean(vec[8:11])

# New IRanges
ir = IRanges(start = 1:5, width=3)
coverage(ir)

# Slice the IRanges "rl" to obtain all the position that is greater than or equal to 2
slice(rl, 2)

# view position 2-8 in IRanges "rl"
vi = Views(rl, IRanges(2,8))

# view position 2-3 & 8-9 in IRanges "rl"
vi = Views(rl, IRanges(c(2,8),width=2))

# The output of "IRanges(c(2,8),width=2)" is as follows:
#IRanges object with 2 ranges and 0 metadata columns:
#          start       end     width
#      <integer> <integer> <integer>
#  [1]         2         3         2
#  [2]         8         9         2


# Calculate mean of ranges listed in "vi"
# vi looks like:
#Views on a 14-length Rle subject
#views:
#    start end width
#[1]     2   3     2 [1 1]
#[2]     8   9     2 [2 2]

mean(vi)

# The output of "mean(vi)": [1] 1 2

# Still problem with the following code .....
vi = Views(rl, as(GRanges("chr1", ranges = IRanges(3,7)), "RangesList"))	# -> 
# Equivalent to
as.numeric(rl)[3:7]





