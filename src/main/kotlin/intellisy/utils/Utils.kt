package intellisy.utils

fun <T> List<T>.toPair(): Pair<T, T> {
    require (this.size == 2) { "List is not of length 2!" }
    return Pair(this[0], this[1])
}