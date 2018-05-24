#!/usr/bin/ruby

require 'matrix'

size_a = [500000, 500000]
size_b = [500000, 600000]

# Probability of space being filled
filled = 0.5

# Range of values
minv = -1.0
maxv = 1.0

if size_a[1] != size_b[0] then
    puts "Matrices of size #{size_a[0]}x#{size_a[1]} and #{size_b[0]}x#{size_b[1]} can't be multiplied!"
    exit -1
end

class Matrix
  def []=(i, j, x)
    @rows[i][j] = x
  end
end


A = Matrix.zero(size_a[0], size_a[1])
B = Matrix.zero(size_b[0], size_b[1])

File.open('A', 'w+') do |f|
    f.puts "#{A.row_count} #{A.column_count}"
    Matrix.build(A.row_count, A.column_count).each do |i, j|
        if Random.rand() < filled then
            val = Random.rand(maxv - minv) + minv
            A[i, j] = val
            f.puts "#{i + 1} #{j + 1} #{val}"
        end
    end
end

File.open('B', 'w+') do |f|
    f.puts "#{B.row_count} #{B.column_count}"
    Matrix.build(B.row_count, B.column_count).each do |i, j|
        if Random.rand() < filled then
            val = Random.rand(maxv - minv) + minv
            B[i, j] = val
            f.puts "#{i + 1} #{j + 1} #{val}"
        end
    end
end

C = A*B

File.open('C', 'w+') do |f|
    f.puts "#{C.row_count} #{C.column_count}"
    Matrix.build(C.row_count, C.column_count).each do |i, j|
        if C[i, j] != 0 then
            f.puts "#{i + 1} #{j + 1} #{C[i, j]}"
        end
    end
end
