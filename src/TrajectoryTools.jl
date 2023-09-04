module TrajectoryTools

"""
```julia
function get_transition_period(bins::AbstractVector{Float64}, rtime::AbstractVector{Float64}, t0::Float64, t1::Float64) where T <: Real
```
Get the start and end index of the transition period for each trial
"""
function get_transition_period(bins::AbstractVector{Float64}, rtime::AbstractVector{Float64}, t0::Float64, t1::Float64)
	ntrials = length(rtime)
	qidx = fill(0, 2, ntrials)
	idx0 = searchsortedfirst(bins, t0)
	qidx[1,:] .= idx0
	for i in 1:ntrials
		rt = rtime[i]
		idx1 = searchsortedlast(bins, rt-t1)
		qidx[2,i] = idx1
	end
	qidx
end

"""
```julia
function compute_triangular_path_length(traj::Matrix{Float64})

Find the largest combined Euclidean distance from the first point to a point on the trajectory, and from that point to the last point.
```
"""
function compute_triangular_path_length(traj::Matrix{T},method=:normal, do_shuffle=false) where T <: Real
	nn = size(traj,1)
	dm = -Inf
	qidx = 0
	if method == :normal
		func = compute_triangular_path_length
	elseif  method == :sq
		func = compute_triangular_path_length_sq
	else
		error("Unknown method $method")
	end
	for i in 2:nn-1
		_d = func(traj, i)
		if _d > dm
			dm = _d
			qidx = i
		end
	end
	dm,qidx
end

"""
```
function compute_triangular_path_length2(traj::Matrix{Float64})

Return the path length of 3 points that best preserves the overall path length
```
"""
function compute_triangular_path_length2(traj::Matrix{T}) where T <: Real
	nn = size(traj,1)
	pl = sum(sqrt.(sum(abs2, diff(traj,dims=1),dims=2)))
	qidx = 0
	dm = Inf
	dq = 0.0
	for i in 2:nn-1
		dd = compute_triangular_path_length(traj, i)
		_dm = dd - pl
		_dm *= _dm
		if _dm < dm
			dm = _dm
			qidx = i
			dq = dd
		end
	end
	dq,qidx
end

function compute_triangular_path_length(traj::Matrix{T},i::Int64) where T <: Real 
	_d = sqrt(sum(abs2, traj[i,:]  - traj[1,:]))
	_d += sqrt(sum(abs2, traj[i,:] - traj[end,:]))
end

"""
```
function compute_triangular_path_length_sq(traj::Matrix{Float64},i::Int64)
````
Computes path length by summing up the square of the line-elements.
"""
function compute_triangular_path_length_sq(traj::Matrix{T},i::Int64) where T <: Real
	_d = sum(abs2, traj[i,:]  - traj[1,:])
	_d += sum(abs2, traj[i,:] - traj[end,:])
end

function compute_triangular_path_length(X::Array{T,3}, qidx::Matrix{Int64}, args...;do_shuffle=false) where T <: Real
	nn = qidx[2,:] - qidx[1,:] .+ 1
	path_lengths = fill(0.0, length(nn))
	for i in 1:length(path_lengths)
		_X = X[qidx[1,i]:qidx[2,i],i,:]
		if do_shuffle
			bidx = shuffle(1:nn[i])	
			_X = _X[bidx,:]
		end
		path_lengths[i],_ = compute_triangular_path_length(_X,args...)
	end
	path_lengths
end

function compute_triangular_path_length2(X::Array{T,3}, qidx::Matrix{Int64}) where T <: Real
	nn = qidx[2,:] - qidx[1,:] .+ 1
	path_lengths = fill(0.0, length(nn))
	for i in 1:length(path_lengths)
		path_lengths[i],_ = compute_triangular_path_length2(X[qidx[1,i]:qidx[2,i],i,:])
	end
	path_lengths
end

"""
```
function compute_ref_path_length(X::Array{T,3}, qidx::Matrix{Int64};do_shuffle=false) where T <: Real
```
Compute path length by using the point closest to the point on a reference trajectory with the highest
energy as a reference point. The reference trajectory is the longest trajectory, as determined by `qidx`.

For each trajectory, find the point closest to the reference point on the reference poin on the reference
trajectory, and then compute the total euclidean distance between this point and the first and last point
of the trajectory, respectively.
"""
function compute_ref_path_length(X::Array{T,3}, qidx::Matrix{Int64};do_shuffle=false) where T <: Real
	epts = get_reference_points(X,qidx)
	# compute triangular path using these points
	pl = fill(0.0, size(X,2))
	for i in axes(pl,1)
		qq = qidx[:,i]
		_X = X[qq[1]:qq[2],i,:] 
		pl[i] = compute_triangular_path_length(_X, epts[i]-qq[1]+1)
	end
	pl
end

function get_reference_points(X::Array{T,3}, qidx::Matrix{Int64};do_shuffle=false) where T <: Real
	# identify the longest trajectory
	nn = qidx[2,:] - qidx[1,:] .+ 1
	refidx = argmax(nn)
	# identify the point on the reference trajectory with the highest energy
	_X = X[qidx[1,refidx]:qidx[2,refidx], refidx, :]
	if do_shuffle
		_X = _X[shuffle(1:size(_X,1)),:]
	end
	eidx = argmax(dropdims(sum(abs2, _X,dims=2),dims=2))

	xr = X[qidx[1,refidx]+eidx-1,refidx,:]
	epts = fill(0, size(X,2))
	# for every other trajectory, find the point closets to the point of maximum energy on the reference
	for i in axes(X,2)
		dm = Inf
		bidx = [qidx[1,i]:qidx[2,i];]
		if do_shuffle
			shuffle!(bidx)
		end
		for j in bidx
			d = sum(abs2, X[j,i,:] - xr)
			if d < dm
				dm = d
				epts[i] = j
			end
		end
	end
	epts
end

function get_ref_pairwise_distance(X::Array{T,3}, qidx::Matrix{Int64}) where T <: Real
	epts = get_reference_points(X,qidx)
	_, nt,nc = size(X)
	D = fill(0.0, nt,nt)
	for i in 1:nt
		for j in 1:nt
			D[j,i] = sqrt(sum(abs2, X[epts[i],i,:] - X[epts[j],j,:]))
		end
	end
	D./nc
end
end # module TrajectoryTools
