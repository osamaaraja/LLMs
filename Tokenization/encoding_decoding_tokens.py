
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

text = """IT was seven o'clock of a very warm evening in the Seeonee hills when
Father Wolf woke up from his day's rest, scratched himself, yawned, and
spread out his paws one after the other to get rid of the sleepy feeling
in the tips. Mother Wolf lay with her big gray nose dropped across her
four tumbling, squealing cubs, and the moon shone into the mouth of the
cave where they all lived. "Augrh!" said Father Wolf, "it is time to
hunt again"; and he was going to spring downhill when a little shadow
with a bushy tail crossed the threshold and whined: "Good luck go with
you, O Chief of the Wolves; and good luck and strong white teeth go with
the noble children, that they may never forget the hungry in this
world."

   [Illustration: "'GOOD LUCK GO WITH YOU, O CHIEF OF THE WOLVES.'"]

It was the jackal--Tabaqui, the Dish-licker--and the wolves of India
despise Tabaqui because he runs about making mischief, and telling
tales, and eating rags and pieces of leather from the village
rubbish-heaps. They are afraid of him too, because Tabaqui, more than
any one else in the jungle, is apt to go mad, and then he forgets that
he was ever afraid of any one, and runs through the forest biting
everything in his way. Even the tiger hides when little Tabaqui goes
mad, for madness is the most disgraceful thing that can overtake a wild
creature. We call it hydrophobia, but they call it _dewanee_--the
madness--and run.

"Enter, then, and look," said Father Wolf, stiffly; "but there is no
food here."

"For a wolf, no," said Tabaqui; "but for so mean a person as myself a
dry bone is a good feast. Who are we, the Gidur-log [the Jackal People],
to pick and choose?" He scuttled to the back of the cave, where he found
the bone of a buck with some meat on it, and sat cracking the end
merrily.

"All thanks for this good meal," he said, licking his lips. "How
beautiful are the noble children! How large are their eyes! And so
young too! Indeed, indeed, I might have remembered that the children of
kings are men from the beginning."

Now, Tabaqui knew as well as any one else that there is nothing so
unlucky as to compliment children to their faces; and it pleased him to
see Mother and Father Wolf look uncomfortable.

Tabaqui sat still, rejoicing in the mischief that he had made, and then
he said spitefully:

"Shere Khan, the Big One, has shifted his hunting-grounds. He will hunt
among these hills during the next moon, so he has told me."

Shere Khan was the tiger who lived near the Waingunga River, twenty
miles away.

"He has no right!" Father Wolf began angrily. "By the Law of the Jungle
he has no right to change his quarters without fair warning. He will
frighten every head of game within ten miles; and I--I have to kill for
two, these days."

"His mother did not call him Lungri [the Lame One] for nothing," said
Mother Wolf, quietly. "He has been lame in one foot from his birth. That
is why he has only killed cattle. Now the villagers of the Waingunga are
angry with him, and he has come here to make _our_ villagers angry.
They will scour the jungle for him when he is far away, and we and our
children must run when the grass is set alight. Indeed, we are very
grateful to Shere Khan!"

"Shall I tell him of your gratitude?" said Tabaqui.

"Out!" snapped Father Wolf. "Out, and hunt with thy master. Thou hast
done harm enough for one night."

"I go," said Tabaqui, quietly. "Ye can hear Shere Khan below in the
thickets. I might have saved myself the message."

Father Wolf listened, and in the dark valley that ran down to a little
river, he heard the dry, angry, snarly, singsong whine of a tiger who
has caught nothing and does not care if all the jungle knows it.

"The fool!" said Father Wolf. "To begin a night's work with that noise!
Does he think that our buck are like his fat Waingunga bullocks?"

"H'sh! It is neither bullock nor buck that he hunts to-night," said
Mother Wolf; "it is Man." The whine had changed to a sort of humming
purr that seemed to roll from every quarter of the compass. It was the
noise that bewilders wood-cutters, and gipsies sleeping in the open,
and makes them run sometimes into the very mouth of the tiger.

"Man!" said Father Wolf, showing all his white teeth. "Faugh! Are there
not enough beetles and frogs in the tanks that he must eat Man--and on
our ground too!"

The Law of the Jungle, which never orders anything without a reason,
forbids every beast to eat Man except when he is killing to show his
children how to kill, and then he must hunt outside the hunting-grounds
of his pack or tribe. The real reason for this is that man-killing
means, sooner or later, the arrival of white men on elephants, with
guns, and hundreds of brown men with gongs and rockets and torches. Then
everybody in the jungle suffers. The reason the beasts give among
themselves is that Man is the weakest and most defenseless of all living
things, and it is unsportsmanlike to touch him. They say too--and it is
true--that man-eaters become mangy, and lose their teeth.

The purr grew louder, and ended in the full-throated "Aaarh!" of the
tiger's charge.

Then there was a howl--an untigerish howl--from Shere Khan. "He has
missed," said Mother Wolf. "What is it?"

Father Wolf ran out a few paces and heard Shere Khan muttering and
mumbling savagely, as he tumbled about in the scrub.

"The fool has had no more sense than to jump at a wood-cutters'
camp-fire, so he has burned his feet," said Father Wolf, with a grunt.
"Tabaqui is with him."

"Something is coming uphill," said Mother Wolf, twitching one ear. "Get
ready."

The bushes rustled a little in the thicket, and Father Wolf dropped with
his haunches under him, ready for his leap. Then, if you had been
watching, you would have seen the most wonderful thing in the world--the
wolf checked in mid-spring. He made his bound before he saw what it was
he was jumping at, and then he tried to stop himself. The result was
that he shot up straight into the air for four or five feet, landing
almost where he left ground.

"Man!" he snapped. "A man's cub. Look!"

Directly in front of him, holding on by a low branch, stood a naked
brown baby who could just walk--as soft and as dimpled a little thing as
ever came to a wolf's cave at night. He looked up into Father Wolf's
face and laughed.

"Is that a man's cub?" said Mother Wolf. "I have never seen one. Bring
it here."

A wolf accustomed to moving his own cubs can, if necessary, mouth an
egg without breaking it, and though Father Wolf's jaws closed right on
the child's back not a tooth even scratched the skin, as he laid it down
among the cubs.

"How little! How naked, and--how bold!" said Mother Wolf, softly. The
baby was pushing his way between the cubs to get close to the warm hide.
"Ahai! He is taking his meal with the others. And so this is a man's
cub. Now, was there ever a wolf that could boast of a man's cub among
her children?"

"I have heard now and again of such a thing, but never in our pack or in
my time," said Father Wolf. "He is altogether without hair, and I could
kill him with a touch of my foot. But see, he looks up and is not
afraid."""

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

vocab_size = 276 # this is set so that we do exactly 20 merges
num_merges = vocab_size - 256

ids = list(tokens)
merges = {}

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    count = stats[pair]
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)} #  vocab is a dict from the token id to the bytes object
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1] # this is addition of two bytes

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids) # b"" concatenates all the bytes together
    # the token above a raw bytes and hence we need to decode
    text = tokens.decode("utf-8", errors='replace') # performing decoding on the bytes object to get the string
    return text

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens) # will count how many times a single pair occurs in the sequence of tokens
        pair = min(stats, key=lambda p: stats.get(p, float("inf")))
        if pair not in merges:
            break # nothing can be merged

        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print(decode(encode("hello world!")))

new_text = decode(encode(text))
if new_text == text:
    print("True")
